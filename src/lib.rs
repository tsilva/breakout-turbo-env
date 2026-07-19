use numpy::{
    PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

const RAW_W: usize = 96;
const RAW_H: usize = 96;
const RENDER_W: usize = 160;
const RENDER_H: usize = 210;
const FP: i32 = 1 << 16;
const BALL_R: i32 = FP;
const PADDLE_W: i32 = 18 * FP;
const PADDLE_H: i32 = 2 * FP;
const PADDLE_Y: i32 = 90 * FP;
const PADDLE_SPEED: i32 = 3 * FP;
const BALL_SPEED_X: i32 = FP / 2;
const BALL_SPEED_Y: i32 = FP;
const BRICK_COLS: usize = 8;
const BRICK_ROWS: usize = 6;
const BRICK_ROW_POINTS: [i32; BRICK_ROWS] = [7, 7, 4, 4, 1, 1];
const BRICK_W: i32 = 10 * FP;
const BRICK_H: i32 = 4 * FP;
const BRICK_GAP_X: i32 = 1 * FP;
const BRICK_GAP_Y: i32 = 1 * FP;
const BRICK_X0: i32 = 4 * FP;
const BRICK_Y0: i32 = 7 * FP;
const FULL_BRICKS: u64 = (1u64 << (BRICK_COLS * BRICK_ROWS)) - 1;
const SIGNALS: usize = 13;

const COLLISION_WALL: i64 = 1;
const COLLISION_PADDLE: i64 = 2;
const COLLISION_BRICK: i64 = 4;
const COLLISION_LOSS: i64 = 8;
const COLLISION_CLEAR: i64 = 16;

#[derive(Clone, Copy)]
struct FastAreaPixel {
    indices: [u16; 4],
    count: u8,
}

#[derive(Clone)]
struct Preprocess {
    out_h: usize,
    out_w: usize,
    crop: [usize; 4], // top, bottom, left, right
    mask_crop: bool,
    crop_fill: u8,
    rows: Vec<(usize, usize)>,
    columns: Vec<(usize, usize)>,
    fast_area: Option<Vec<FastAreaPixel>>,
    fast_two_by_two: bool,
}

#[derive(Clone)]
struct Lane {
    paddle_x: i32,
    ball_x: i32,
    ball_y: i32,
    ball_vx: i32,
    ball_vy: i32,
    bricks: u64,
    score: i32,
    lives: i32,
    tick: u64,
    layout_id: i32,
    pending_reset: bool,
    last_collision: i64,
    stack: Vec<u8>,
    stack_head: usize,
    raw_scratch: Vec<u8>,
    brick_layer: Vec<u8>,
    rendered_paddle_x: usize,
    rendered_ball_x: usize,
    rendered_ball_y: usize,
    render_initialized: bool,
}

impl Lane {
    fn new(stack_size: usize) -> Self {
        Self {
            paddle_x: ((RAW_W as i32 * FP) - PADDLE_W) / 2,
            ball_x: RAW_W as i32 * FP / 2,
            ball_y: 82 * FP,
            ball_vx: BALL_SPEED_X,
            ball_vy: -BALL_SPEED_Y,
            bricks: FULL_BRICKS,
            score: 0,
            lives: 3,
            tick: 0,
            layout_id: 0,
            pending_reset: false,
            last_collision: 0,
            stack: vec![0; stack_size],
            stack_head: 0,
            raw_scratch: vec![0; RAW_W * RAW_H],
            brick_layer: vec![0; RAW_W * RAW_H],
            rendered_paddle_x: 0,
            rendered_ball_x: 0,
            rendered_ball_y: 0,
            render_initialized: false,
        }
    }
}

fn layout_mask(layout_id: i32) -> Option<u64> {
    match layout_id {
        0 => Some(FULL_BRICKS),
        1 => {
            let mut mask = 0u64;
            for row in 0..BRICK_ROWS {
                for col in 0..BRICK_COLS {
                    if (row + col) % 2 == 0 {
                        mask |= 1u64 << (row * BRICK_COLS + col);
                    }
                }
            }
            Some(mask)
        }
        2 => {
            let mut mask = FULL_BRICKS;
            for row in 1..BRICK_ROWS {
                mask &= !(1u64 << (row * BRICK_COLS + 3));
                mask &= !(1u64 << (row * BRICK_COLS + 4));
            }
            Some(mask)
        }
        3 => {
            let mut mask = 0u64;
            for col in 0..BRICK_COLS {
                mask |= 1u64 << col;
                mask |= 1u64 << ((BRICK_ROWS - 1) * BRICK_COLS + col);
            }
            Some(mask)
        }
        _ => None,
    }
}

fn reset_lane(lane: &mut Lane, layout_id: i32, preprocess: &Preprocess, frame_stack: usize) {
    lane.paddle_x = ((RAW_W as i32 * FP) - PADDLE_W) / 2;
    lane.ball_x = RAW_W as i32 * FP / 2;
    lane.ball_y = 82 * FP;
    lane.ball_vx = BALL_SPEED_X;
    lane.ball_vy = -BALL_SPEED_Y;
    lane.bricks = layout_mask(layout_id).expect("validated layout");
    lane.score = 0;
    lane.lives = 3;
    lane.tick = 0;
    lane.layout_id = layout_id;
    lane.pending_reset = false;
    lane.last_collision = 0;
    lane.stack.fill(0);
    lane.stack_head = 0;
    lane.render_initialized = false;
    let plane = preprocess.out_h * preprocess.out_w;
    render_and_push(lane, preprocess, frame_stack);
    for slot in 1..frame_stack {
        lane.stack.copy_within(0..plane, slot * plane);
    }
    lane.stack_head = 0;
}

fn step_native(lane: &mut Lane, action: u8) -> (f32, bool) {
    let score_before = lane.score;
    lane.last_collision = 0;
    match action {
        1 => lane.paddle_x -= PADDLE_SPEED,
        2 => lane.paddle_x += PADDLE_SPEED,
        _ => {}
    }
    lane.paddle_x = lane.paddle_x.clamp(0, RAW_W as i32 * FP - PADDLE_W);

    // Fixed microsteps make collision behavior deterministic and prevent tunnelling.
    const MICROSTEPS: i32 = 4;
    for _ in 0..MICROSTEPS {
        let previous_x = lane.ball_x;
        let previous_y = lane.ball_y;
        lane.ball_x += lane.ball_vx / MICROSTEPS;
        lane.ball_y += lane.ball_vy / MICROSTEPS;

        if lane.ball_x - BALL_R <= 0 {
            lane.ball_x = BALL_R;
            lane.ball_vx = lane.ball_vx.abs();
            lane.last_collision |= COLLISION_WALL;
        } else if lane.ball_x + BALL_R >= RAW_W as i32 * FP {
            lane.ball_x = RAW_W as i32 * FP - BALL_R;
            lane.ball_vx = -lane.ball_vx.abs();
            lane.last_collision |= COLLISION_WALL;
        }
        if lane.ball_y - BALL_R <= 0 {
            lane.ball_y = BALL_R;
            lane.ball_vy = lane.ball_vy.abs();
            lane.last_collision |= COLLISION_WALL;
        }

        let paddle_right = lane.paddle_x + PADDLE_W;
        if lane.ball_vy > 0
            && lane.ball_y + BALL_R >= PADDLE_Y
            && lane.ball_y - BALL_R <= PADDLE_Y + PADDLE_H
            && lane.ball_x + BALL_R >= lane.paddle_x
            && lane.ball_x - BALL_R <= paddle_right
        {
            lane.ball_y = PADDLE_Y - BALL_R;
            lane.ball_vy = -lane.ball_vy.abs();
            let paddle_center = lane.paddle_x + PADDLE_W / 2;
            let offset = lane.ball_x - paddle_center;
            lane.ball_vx = (BALL_SPEED_X + offset / 8).clamp(-2 * FP, 2 * FP);
            if lane.ball_vx == 0 {
                lane.ball_vx = if offset < 0 { -FP / 2 } else { FP / 2 };
            }
            lane.last_collision |= COLLISION_PADDLE;
        }

        if let Some(index) = colliding_brick(lane.ball_x, lane.ball_y, lane.bricks) {
            lane.bricks &= !(1u64 << index);
            clear_brick_from_render_cache(lane, index);
            bounce_from_brick(lane, index, previous_x, previous_y);
            lane.score += BRICK_ROW_POINTS[index / BRICK_COLS];
            lane.last_collision |= COLLISION_BRICK;
        }

        if lane.ball_y - BALL_R >= RAW_H as i32 * FP {
            lane.lives -= 1;
            lane.last_collision |= COLLISION_LOSS;
            if lane.lives <= 0 {
                lane.pending_reset = true;
                lane.tick += 1;
                return ((lane.score - score_before) as f32, true);
            }
            lane.ball_x = RAW_W as i32 * FP / 2;
            lane.ball_y = 82 * FP;
            lane.ball_vx = BALL_SPEED_X;
            lane.ball_vy = -BALL_SPEED_Y;
            break;
        }
    }
    lane.tick += 1;
    if lane.bricks == 0 {
        lane.last_collision |= COLLISION_CLEAR;
        lane.pending_reset = true;
        return ((lane.score - score_before) as f32, true);
    }
    ((lane.score - score_before) as f32, false)
}

fn colliding_brick(x: i32, y: i32, mask: u64) -> Option<usize> {
    let cell_w = BRICK_W + BRICK_GAP_X;
    let cell_h = BRICK_H + BRICK_GAP_Y;
    let min_x = x - BALL_R - BRICK_X0;
    let max_x = x + BALL_R - BRICK_X0;
    let min_y = y - BALL_R - BRICK_Y0;
    let max_y = y + BALL_R - BRICK_Y0;
    if max_x < 0
        || max_y < 0
        || min_x > BRICK_COLS as i32 * cell_w
        || min_y > BRICK_ROWS as i32 * cell_h
    {
        return None;
    }
    let min_col = (min_x.max(0) / cell_w).min(BRICK_COLS as i32 - 1) as usize;
    let max_col = (max_x.max(0) / cell_w).min(BRICK_COLS as i32 - 1) as usize;
    let min_row = (min_y.max(0) / cell_h).min(BRICK_ROWS as i32 - 1) as usize;
    let max_row = (max_y.max(0) / cell_h).min(BRICK_ROWS as i32 - 1) as usize;
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            let index = row * BRICK_COLS + col;
            if (mask >> index) & 1 == 0 {
                continue;
            }
            let left = BRICK_X0 + col as i32 * (BRICK_W + BRICK_GAP_X);
            let top = BRICK_Y0 + row as i32 * (BRICK_H + BRICK_GAP_Y);
            let right = left + BRICK_W;
            let bottom = top + BRICK_H;
            if x + BALL_R >= left
                && x - BALL_R <= right
                && y + BALL_R >= top
                && y - BALL_R <= bottom
            {
                return Some(index);
            }
        }
    }
    None
}

fn bounce_from_brick(lane: &mut Lane, index: usize, previous_x: i32, previous_y: i32) {
    let row = index / BRICK_COLS;
    let col = index % BRICK_COLS;
    let left = BRICK_X0 + col as i32 * (BRICK_W + BRICK_GAP_X);
    let top = BRICK_Y0 + row as i32 * (BRICK_H + BRICK_GAP_Y);
    let right = left + BRICK_W;
    let bottom = top + BRICK_H;

    if previous_y + BALL_R <= top {
        lane.ball_y = top - BALL_R;
        lane.ball_vy = -lane.ball_vy.abs();
    } else if previous_y - BALL_R >= bottom {
        lane.ball_y = bottom + BALL_R;
        lane.ball_vy = lane.ball_vy.abs();
    } else if previous_x + BALL_R <= left {
        lane.ball_x = left - BALL_R;
        lane.ball_vx = -lane.ball_vx.abs();
    } else if previous_x - BALL_R >= right {
        lane.ball_x = right + BALL_R;
        lane.ball_vx = lane.ball_vx.abs();
    } else {
        let horizontal_penetration =
            (lane.ball_x + BALL_R - left).min(right - (lane.ball_x - BALL_R));
        let vertical_penetration =
            (lane.ball_y + BALL_R - top).min(bottom - (lane.ball_y - BALL_R));
        if vertical_penetration <= horizontal_penetration {
            lane.ball_vy = -lane.ball_vy;
        } else {
            lane.ball_vx = -lane.ball_vx;
        }
    }
}

const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b100, 0b100, 0b100, 0b100, 0b100],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b001, 0b001, 0b001],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

fn draw_digit(frame: &mut [u8], digit: usize, x0: usize) {
    for (row, bits) in DIGITS[digit % 10].iter().enumerate() {
        for col in 0..3 {
            if bits & (1 << (2 - col)) == 0 {
                continue;
            }
            for y in 5 + row * 2..5 + row * 2 + 2 {
                frame[y * RENDER_W + x0 + col * 4..y * RENDER_W + x0 + col * 4 + 4].fill(1);
            }
        }
    }
}

fn render_ball_y(source_y: usize) -> usize {
    match source_y {
        0..=7 => 31 + source_y * 26 / 7,
        8..=35 => 57 + (source_y - 7) * 36 / 28,
        36..=90 => 93 + (source_y - 35) * 96 / 55,
        _ => 189 + (source_y - 90) * 20 / 6,
    }
}

fn render_indexed(lane: &Lane) -> Vec<u8> {
    let mut frame = vec![0u8; RENDER_W * RENDER_H];

    // Atari's status display lives inside the video signal: three score
    // digits, the selected game number, and the current player number.
    let score = lane.score.clamp(0, 999) as usize;
    draw_digit(&mut frame, score / 100, 36);
    draw_digit(&mut frame, (score / 10) % 10, 52);
    draw_digit(&mut frame, score % 10, 68);
    draw_digit(&mut frame, (lane.layout_id + 1).clamp(0, 9) as usize, 100);
    draw_digit(&mut frame, 1, 136);

    // Stella's native frame uses a 15-line header bar and 8-pixel
    // playfield walls in the same neutral gray.
    for y in 17..32 {
        frame[y * RENDER_W..(y + 1) * RENDER_W].fill(1);
    }
    for y in 32..189 {
        frame[y * RENDER_W..y * RENDER_W + 8].fill(1);
        frame[y * RENDER_W + 152..(y + 1) * RENDER_W].fill(1);
    }

    // TIA playfield pixels meet edge-to-edge, so a complete wall appears as
    // six uninterrupted color bands. The turbo environment's eight logical
    // brick columns divide the same 144-pixel playfield without adding gaps.
    for row in 0..BRICK_ROWS {
        for col in 0..BRICK_COLS {
            let index = row * BRICK_COLS + col;
            if (lane.bricks >> index) & 1 == 0 {
                continue;
            }
            let x0 = 8 + col * 18;
            let y0 = 57 + row * 6;
            for y in y0..y0 + 6 {
                frame[y * RENDER_W + x0..y * RENDER_W + x0 + 18].fill(2 + row as u8);
            }
        }
    }

    let source_paddle_x = (lane.paddle_x / FP).clamp(0, 78) as usize;
    // Stella's paddle travels from x=8 through x=144. At the right limit its
    // final eight pixels merge with the fixed red edge cap.
    let paddle_x = 8 + source_paddle_x * 136 / 78;
    for y in 189..193 {
        frame[y * RENDER_W + paddle_x..y * RENDER_W + paddle_x + 16].fill(2);
    }

    let source_ball_x = (lane.ball_x / FP).clamp(1, RAW_W as i32 - 1) as usize;
    let source_ball_y = (lane.ball_y / FP).clamp(0, RAW_H as i32 - 1) as usize;
    // Preserve all three Stella anchors: left x=8, launch x=80, and right
    // x=150. The physical arena is asymmetric around its integer launch
    // center, so each half needs its own linear projection.
    let ball_x = if source_ball_x <= 48 {
        8 + ((source_ball_x - 1) * 72 + 46) / 47
    } else {
        80 + ((source_ball_x - 48) * 70 + 23) / 47
    };
    let ball_y = render_ball_y(source_ball_y)
        .saturating_sub(2)
        .min(RENDER_H - 4);
    for y in ball_y..ball_y + 4 {
        frame[y * RENDER_W + ball_x..y * RENDER_W + ball_x + 2].fill(2);
    }

    // The original four-paddle kernel leaves the inactive players clipped at
    // the lower wall edges in Stella's output.
    for y in 189..196 {
        frame[y * RENDER_W..y * RENDER_W + 8].fill(8);
    }
    for y in 189..195 {
        frame[y * RENDER_W + 152..(y + 1) * RENDER_W].fill(2);
    }
    frame
}

fn palette_gray(index: u8) -> u8 {
    match index {
        0 => 0,
        1 => 180,
        2 => 255,
        3 => 72,
        4 => 96,
        5 => 120,
        6 => 144,
        7 => 168,
        8 => 192,
        _ => 0,
    }
}

fn initialize_render_cache(lane: &mut Lane) {
    lane.brick_layer.fill(0);
    for row in 0..BRICK_ROWS {
        for col in 0..BRICK_COLS {
            let index = row * BRICK_COLS + col;
            if (lane.bricks >> index) & 1 == 0 {
                continue;
            }
            let x0 = 4 + col * 11;
            let y0 = 7 + row * 5;
            let value = palette_gray(3 + row as u8);
            for y in y0..(y0 + 4).min(RAW_H) {
                lane.brick_layer[y * RAW_W + x0..y * RAW_W + (x0 + 10).min(RAW_W)].fill(value);
            }
        }
    }
    lane.raw_scratch.copy_from_slice(&lane.brick_layer);
    lane.render_initialized = true;
}

fn clear_brick_from_render_cache(lane: &mut Lane, index: usize) {
    if !lane.render_initialized {
        return;
    }
    let row = index / BRICK_COLS;
    let col = index % BRICK_COLS;
    let x0 = 4 + col * 11;
    let y0 = 7 + row * 5;
    for y in y0..(y0 + 4).min(RAW_H) {
        let begin = y * RAW_W + x0;
        let end = y * RAW_W + (x0 + 10).min(RAW_W);
        lane.brick_layer[begin..end].fill(0);
        lane.raw_scratch[begin..end].fill(0);
    }
}

fn restore_previous_dynamic_pixels(lane: &mut Lane) {
    let paddle_y = (PADDLE_Y / FP) as usize;
    let paddle_h = (PADDLE_H / FP) as usize;
    let paddle_w = (PADDLE_W / FP) as usize;
    let paddle_right = (lane.rendered_paddle_x + paddle_w).min(RAW_W);
    for y in paddle_y..paddle_y + paddle_h {
        let begin = y * RAW_W + lane.rendered_paddle_x;
        let end = y * RAW_W + paddle_right;
        lane.raw_scratch[begin..end].copy_from_slice(&lane.brick_layer[begin..end]);
    }

    let left = lane.rendered_ball_x.saturating_sub(1);
    let right = (lane.rendered_ball_x + 1).min(RAW_W - 1);
    let top = lane.rendered_ball_y.saturating_sub(1);
    let bottom = (lane.rendered_ball_y + 1).min(RAW_H - 1);
    for y in top..=bottom {
        let begin = y * RAW_W + left;
        let end = y * RAW_W + right + 1;
        lane.raw_scratch[begin..end].copy_from_slice(&lane.brick_layer[begin..end]);
    }
}

fn render_and_push(lane: &mut Lane, preprocess: &Preprocess, frame_stack: usize) {
    if lane.render_initialized {
        restore_previous_dynamic_pixels(lane);
    } else {
        initialize_render_cache(lane);
    }
    let px = (lane.paddle_x / FP).clamp(0, RAW_W as i32 - 1) as usize;
    let paddle_y = (PADDLE_Y / FP) as usize;
    let paddle_h = (PADDLE_H / FP) as usize;
    let paddle_w = (PADDLE_W / FP) as usize;
    for y in paddle_y..paddle_y + paddle_h {
        lane.raw_scratch[y * RAW_W + px..y * RAW_W + (px + paddle_w).min(RAW_W)]
            .fill(palette_gray(1));
    }
    let bx = (lane.ball_x / FP).clamp(0, RAW_W as i32 - 1) as usize;
    let by = (lane.ball_y / FP).clamp(0, RAW_H as i32 - 1) as usize;
    for y in by.saturating_sub(1)..=(by + 1).min(RAW_H - 1) {
        lane.raw_scratch[y * RAW_W + bx.saturating_sub(1)..y * RAW_W + (bx + 2).min(RAW_W)]
            .fill(palette_gray(2));
    }
    lane.rendered_paddle_x = px;
    lane.rendered_ball_x = bx;
    lane.rendered_ball_y = by;
    let raw = &mut lane.raw_scratch;
    let [top, bottom, left, right] = preprocess.crop;
    let (source_y, source_x) = if preprocess.mask_crop {
        for y in 0..RAW_H {
            for x in 0..RAW_W {
                if y < top || y >= RAW_H - bottom || x < left || x >= RAW_W - right {
                    raw[y * RAW_W + x] = preprocess.crop_fill;
                }
            }
        }
        (0, 0)
    } else {
        (top, left)
    };
    let plane = preprocess.out_h * preprocess.out_w;
    let destination_slot = lane.stack_head;
    let out = &mut lane.stack[destination_slot * plane..(destination_slot + 1) * plane];
    // Deterministic integer box-area resize. Upsampled axes naturally select a
    // single source pixel; downsampled axes average every covered source pixel.
    if preprocess.fast_two_by_two {
        for (y, &(source_row, _)) in preprocess.rows.iter().enumerate() {
            let row0 = (source_y + source_row) * RAW_W + source_x;
            let row1 = row0 + RAW_W;
            let output_row = y * preprocess.out_w;
            for (x, &(source_column, _)) in preprocess.columns.iter().enumerate() {
                // Construction validates that every area box is exactly 2x2
                // and fully inside the fixed raw frame.
                let value = unsafe {
                    (*raw.get_unchecked(row0 + source_column) as u16
                        + *raw.get_unchecked(row0 + source_column + 1) as u16
                        + *raw.get_unchecked(row1 + source_column) as u16
                        + *raw.get_unchecked(row1 + source_column + 1) as u16)
                        >> 2
                };
                out[output_row + x] = value as u8;
            }
        }
    } else if let Some(plan) = &preprocess.fast_area {
        for (dst, pixel) in out.iter_mut().zip(plan) {
            let indices = pixel.indices;
            *dst = match pixel.count {
                1 => raw[indices[0] as usize],
                2 => {
                    ((raw[indices[0] as usize] as u16 + raw[indices[1] as usize] as u16) >> 1) as u8
                }
                4 => {
                    ((raw[indices[0] as usize] as u16
                        + raw[indices[1] as usize] as u16
                        + raw[indices[2] as usize] as u16
                        + raw[indices[3] as usize] as u16)
                        >> 2) as u8
                }
                count => {
                    let sum = indices[..count as usize]
                        .iter()
                        .map(|&index| raw[index as usize] as u16)
                        .sum::<u16>();
                    (sum / count as u16) as u8
                }
            };
        }
    } else {
        for (y, &(sy0, sy1)) in preprocess.rows.iter().enumerate() {
            for (x, &(sx0, sx1)) in preprocess.columns.iter().enumerate() {
                let mut sum = 0usize;
                let mut count = 0usize;
                for sy in sy0..sy1 {
                    for sx in sx0..sx1 {
                        sum += raw[(source_y + sy) * RAW_W + source_x + sx] as usize;
                        count += 1;
                    }
                }
                out[y * preprocess.out_w + x] = (sum / count) as u8;
            }
        }
    }
    lane.stack_head = (lane.stack_head + 1) % frame_stack;
}

fn write_stack(lane: &Lane, dst: &mut [u8], frame_stack: usize) {
    let plane = dst.len() / frame_stack;
    let split = lane.stack_head * plane;
    let tail = lane.stack.len() - split;
    dst[..tail].copy_from_slice(&lane.stack[split..]);
    dst[tail..].copy_from_slice(&lane.stack[..split]);
}

fn write_signals(lane: &Lane, dst: &mut [i64]) {
    dst[0] = lane.paddle_x as i64;
    dst[1] = lane.ball_x as i64;
    dst[2] = lane.ball_y as i64;
    dst[3] = lane.ball_vx as i64;
    dst[4] = lane.ball_vy as i64;
    dst[5] = lane.bricks as i64;
    dst[6] = lane.score as i64;
    dst[7] = lane.lives as i64;
    dst[8] = lane.tick as i64;
    dst[9] = lane.bricks.count_ones() as i64;
    dst[10] = lane.layout_id as i64;
    dst[11] = lane.last_collision;
    dst[12] = lane.pending_reset as i64;
}

fn put_i32(dst: &mut Vec<u8>, value: i32) {
    dst.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(dst: &mut Vec<u8>, value: u64) {
    dst.extend_from_slice(&value.to_le_bytes());
}
fn take_i32(src: &[u8], offset: &mut usize) -> Result<i32, &'static str> {
    let bytes: [u8; 4] = src
        .get(*offset..*offset + 4)
        .ok_or("state is truncated")?
        .try_into()
        .unwrap();
    *offset += 4;
    Ok(i32::from_le_bytes(bytes))
}
fn take_u64(src: &[u8], offset: &mut usize) -> Result<u64, &'static str> {
    let bytes: [u8; 8] = src
        .get(*offset..*offset + 8)
        .ok_or("state is truncated")?
        .try_into()
        .unwrap();
    *offset += 8;
    Ok(u64::from_le_bytes(bytes))
}

fn serialize_lane(lane: &Lane) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + lane.stack.len());
    out.extend_from_slice(b"BTO1");
    for value in [
        lane.paddle_x,
        lane.ball_x,
        lane.ball_y,
        lane.ball_vx,
        lane.ball_vy,
        lane.score,
        lane.lives,
        lane.layout_id,
    ] {
        put_i32(&mut out, value);
    }
    put_u64(&mut out, lane.bricks);
    put_u64(&mut out, lane.tick);
    put_u64(&mut out, lane.last_collision as u64);
    put_u64(&mut out, lane.stack_head as u64);
    out.push(lane.pending_reset as u8);
    out.extend_from_slice(&lane.stack);
    out
}

fn deserialize_lane(data: &[u8], expected_stack: usize) -> Result<Lane, &'static str> {
    if data.get(0..4) != Some(b"BTO1") {
        return Err("state has an invalid header");
    }
    let mut offset = 4;
    let paddle_x = take_i32(data, &mut offset)?;
    let ball_x = take_i32(data, &mut offset)?;
    let ball_y = take_i32(data, &mut offset)?;
    let ball_vx = take_i32(data, &mut offset)?;
    let ball_vy = take_i32(data, &mut offset)?;
    let score = take_i32(data, &mut offset)?;
    let lives = take_i32(data, &mut offset)?;
    let layout_id = take_i32(data, &mut offset)?;
    let bricks = take_u64(data, &mut offset)?;
    let tick = take_u64(data, &mut offset)?;
    let last_collision = take_u64(data, &mut offset)? as i64;
    let stack_head = take_u64(data, &mut offset)? as usize;
    let pending_reset = *data.get(offset).ok_or("state is truncated")? != 0;
    offset += 1;
    let stack = data.get(offset..).ok_or("state is truncated")?.to_vec();
    if stack.len() != expected_stack {
        return Err("state observation shape does not match this environment");
    }
    Ok(Lane {
        paddle_x,
        ball_x,
        ball_y,
        ball_vx,
        ball_vy,
        bricks,
        score,
        lives,
        tick,
        layout_id,
        pending_reset,
        last_collision,
        stack,
        stack_head,
        raw_scratch: vec![0; RAW_W * RAW_H],
        brick_layer: vec![0; RAW_W * RAW_H],
        rendered_paddle_x: 0,
        rendered_ball_x: 0,
        rendered_ball_y: 0,
        render_initialized: false,
    })
}

#[pyclass]
struct NativeBreakoutVecEnv {
    lanes: Vec<Lane>,
    obs_h: usize,
    obs_w: usize,
    frame_skip: usize,
    frame_stack: usize,
    pool: rayon::ThreadPool,
    preprocess: Preprocess,
}

#[pymethods]
impl NativeBreakoutVecEnv {
    #[new]
    fn new(
        num_envs: usize,
        obs_h: usize,
        obs_w: usize,
        frame_skip: usize,
        frame_stack: usize,
        num_threads: usize,
        crop: Vec<usize>,
        mask_crop: bool,
        crop_fill: u8,
    ) -> PyResult<Self> {
        if num_envs == 0 || obs_h == 0 || obs_w == 0 || frame_skip == 0 || frame_stack == 0 {
            return Err(PyValueError::new_err(
                "num_envs, observation dimensions, frame_skip, and frame_stack must be positive",
            ));
        }
        if crop.len() != 4 || crop[0] + crop[1] >= RAW_H || crop[2] + crop[3] >= RAW_W {
            return Err(PyValueError::new_err(
                "crop must contain top, bottom, left, right and preserve at least one source pixel",
            ));
        }
        let source_h = if mask_crop {
            RAW_H
        } else {
            RAW_H - crop[0] - crop[1]
        };
        let source_w = if mask_crop {
            RAW_W
        } else {
            RAW_W - crop[2] - crop[3]
        };
        let rows: Vec<(usize, usize)> = (0..obs_h)
            .map(|y| {
                let begin = y * source_h / obs_h;
                let end = (((y + 1) * source_h + obs_h - 1) / obs_h)
                    .max(begin + 1)
                    .min(source_h);
                (begin, end)
            })
            .collect();
        let columns: Vec<(usize, usize)> = (0..obs_w)
            .map(|x| {
                let begin = x * source_w / obs_w;
                let end = (((x + 1) * source_w + obs_w - 1) / obs_w)
                    .max(begin + 1)
                    .min(source_w);
                (begin, end)
            })
            .collect();
        let source_y = if mask_crop { 0 } else { crop[0] };
        let source_x = if mask_crop { 0 } else { crop[2] };
        let fast_area = if rows.iter().all(|&(begin, end)| end - begin <= 2)
            && columns.iter().all(|&(begin, end)| end - begin <= 2)
        {
            let mut plan = Vec::with_capacity(obs_h * obs_w);
            for &(row_begin, row_end) in &rows {
                for &(column_begin, column_end) in &columns {
                    let mut indices = [0u16; 4];
                    let mut count = 0usize;
                    for row in row_begin..row_end {
                        for column in column_begin..column_end {
                            indices[count] = ((source_y + row) * RAW_W + source_x + column) as u16;
                            count += 1;
                        }
                    }
                    plan.push(FastAreaPixel {
                        indices,
                        count: count as u8,
                    });
                }
            }
            Some(plan)
        } else {
            None
        };
        let fast_two_by_two = rows.iter().all(|&(begin, end)| end - begin == 2)
            && columns.iter().all(|&(begin, end)| end - begin == 2);
        let preprocess = Preprocess {
            out_h: obs_h,
            out_w: obs_w,
            crop: [crop[0], crop[1], crop[2], crop[3]],
            mask_crop,
            crop_fill,
            rows,
            columns,
            fast_area,
            fast_two_by_two,
        };
        let stack_size = obs_h
            .checked_mul(obs_w)
            .and_then(|v| v.checked_mul(frame_stack))
            .ok_or_else(|| PyValueError::new_err("observation size overflow"))?;
        let mut lanes = (0..num_envs)
            .map(|_| Lane::new(stack_size))
            .collect::<Vec<_>>();
        for lane in &mut lanes {
            reset_lane(lane, 0, &preprocess, frame_stack);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.max(1).min(num_envs))
            .build()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            lanes,
            obs_h,
            obs_w,
            frame_skip,
            frame_stack,
            pool,
            preprocess,
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.lanes.len()
    }

    fn reset_into(
        &mut self,
        py: Python<'_>,
        reset_mask: PyReadonlyArray1<'_, bool>,
        start_indices: PyReadonlyArray1<'_, i32>,
        mut observations: PyReadwriteArray4<'_, u8>,
        mut signals: PyReadwriteArray2<'_, i64>,
    ) -> PyResult<()> {
        let mask = reset_mask.as_slice()?;
        let starts = start_indices.as_slice()?;
        self.validate_shapes(
            mask.len(),
            starts.len(),
            observations.shape(),
            signals.shape(),
        )?;
        for (&selected, &start) in mask.iter().zip(starts) {
            if selected && layout_mask(if start < 0 { 0 } else { start }).is_none() {
                return Err(PyValueError::new_err(
                    "selected start_indices must be -1 or an index in [0, 4)",
                ));
            }
        }
        let obs = observations.as_slice_mut()?;
        let signal_data = signals.as_slice_mut()?;
        let obs_per_env = self.frame_stack * self.obs_h * self.obs_w;
        let preprocess = &self.preprocess;
        let frame_stack = self.frame_stack;
        let pool = &self.pool;
        py.allow_threads(|| {
            pool.install(|| {
                self.lanes
                    .par_iter_mut()
                    .zip(mask.par_iter())
                    .zip(starts.par_iter())
                    .zip(obs.par_chunks_mut(obs_per_env))
                    .zip(signal_data.par_chunks_mut(SIGNALS))
                    .for_each(|((((lane, &selected), &start), obs_dst), signal_dst)| {
                        if selected {
                            reset_lane(
                                lane,
                                if start < 0 { 0 } else { start },
                                preprocess,
                                frame_stack,
                            );
                        }
                        write_stack(lane, obs_dst, frame_stack);
                        write_signals(lane, signal_dst);
                    });
            })
        });
        Ok(())
    }

    fn step_into(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray1<'_, u8>,
        mut observations: PyReadwriteArray4<'_, u8>,
        mut rewards: PyReadwriteArray1<'_, f32>,
        mut terminated: PyReadwriteArray1<'_, bool>,
        mut truncated: PyReadwriteArray1<'_, bool>,
        mut signals: PyReadwriteArray2<'_, i64>,
        write_info: bool,
    ) -> PyResult<()> {
        if self.lanes.iter().any(|lane| lane.pending_reset) {
            return Err(PyRuntimeError::new_err(
                "cannot step while a lane is pending reset; reset completed lanes with options={'reset_mask': mask}",
            ));
        }
        let actions = actions.as_slice()?;
        self.validate_step_shapes(
            actions.len(),
            observations.shape(),
            rewards.len(),
            terminated.len(),
            truncated.len(),
            signals.shape(),
        )?;
        if let Some(action) = actions.iter().find(|&&action| action > 2) {
            return Err(PyValueError::new_err(format!(
                "actions must be 0, 1, or 2; got {action}"
            )));
        }
        let obs = observations.as_slice_mut()?;
        let rewards = rewards.as_slice_mut()?;
        let terminated = terminated.as_slice_mut()?;
        let truncated = truncated.as_slice_mut()?;
        let signal_data = signals.as_slice_mut()?;
        let obs_per_env = self.frame_stack * self.obs_h * self.obs_w;
        let preprocess = &self.preprocess;
        let frame_stack = self.frame_stack;
        let frame_skip = self.frame_skip;
        let pool = &self.pool;
        py.allow_threads(|| {
            pool.install(|| {
                self.lanes
                    .par_iter_mut()
                    .zip(actions.par_iter())
                    .zip(obs.par_chunks_mut(obs_per_env))
                    .zip(rewards.par_iter_mut())
                    .zip(terminated.par_iter_mut())
                    .zip(truncated.par_iter_mut())
                    .zip(signal_data.par_chunks_mut(SIGNALS))
                    .for_each(
                        |(
                            (((((lane, &action), obs_dst), reward_dst), term_dst), trunc_dst),
                            signal_dst,
                        )| {
                            let mut reward = 0.0;
                            let mut done = false;
                            let mut collision_events = 0i64;
                            for _ in 0..frame_skip {
                                let (step_reward, step_done) = step_native(lane, action);
                                reward += step_reward;
                                collision_events |= lane.last_collision;
                                if step_done {
                                    done = true;
                                    break;
                                }
                            }
                            lane.last_collision = collision_events;
                            render_and_push(lane, preprocess, frame_stack);
                            write_stack(lane, obs_dst, frame_stack);
                            if write_info {
                                write_signals(lane, signal_dst);
                            }
                            *reward_dst = reward;
                            *term_dst = done;
                            *trunc_dst = false;
                        },
                    );
            })
        });
        Ok(())
    }

    fn get_states(&self) -> Vec<Vec<u8>> {
        self.lanes.iter().map(serialize_lane).collect()
    }

    fn set_states(
        &mut self,
        states: Vec<Vec<u8>>,
        reset_mask: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<()> {
        let mask = reset_mask.as_slice()?;
        if states.len() != self.lanes.len() || mask.len() != self.lanes.len() {
            return Err(PyValueError::new_err(
                "states and reset_mask must have num_envs entries",
            ));
        }
        let expected = self.frame_stack * self.obs_h * self.obs_w;
        let mut replacements = Vec::with_capacity(states.len());
        for (index, state) in states.iter().enumerate() {
            if mask[index] {
                replacements.push(Some(
                    deserialize_lane(state, expected).map_err(PyValueError::new_err)?,
                ));
            } else {
                replacements.push(None);
            }
        }
        for (lane, replacement) in self.lanes.iter_mut().zip(replacements) {
            if let Some(value) = replacement {
                *lane = value;
            }
        }
        Ok(())
    }

    fn render_indexed(&self, lane: usize) -> PyResult<Vec<u8>> {
        self.lanes
            .get(lane)
            .map(render_indexed)
            .ok_or_else(|| PyValueError::new_err("lane index out of range"))
    }

    fn layout_ids(&self) -> Vec<i32> {
        self.lanes.iter().map(|lane| lane.layout_id).collect()
    }

    fn branch(
        &self,
        states: Vec<Vec<u8>>,
        actions: Vec<u8>,
    ) -> PyResult<(Vec<Vec<u8>>, Vec<u8>, Vec<f32>, Vec<bool>, Vec<i64>)> {
        if actions.iter().any(|&action| action > 2) {
            return Err(PyValueError::new_err("actions must be 0, 1, or 2"));
        }
        let expected = self.frame_stack * self.obs_h * self.obs_w;
        let mut base = Vec::with_capacity(states.len());
        for state in &states {
            base.push(deserialize_lane(state, expected).map_err(PyValueError::new_err)?);
        }
        let count = base.len() * actions.len();
        let mut rows = (0..count)
            .map(|index| {
                (
                    base[index / actions.len()].clone(),
                    actions[index % actions.len()],
                    0.0f32,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let preprocess = &self.preprocess;
        let frame_stack = self.frame_stack;
        let frame_skip = self.frame_skip;
        self.pool.install(|| {
            rows.par_iter_mut()
                .for_each(|(lane, action, total_reward, terminated)| {
                    let mut collision_events = 0i64;
                    for _ in 0..frame_skip {
                        let (reward, done) = step_native(lane, *action);
                        *total_reward += reward;
                        collision_events |= lane.last_collision;
                        if done {
                            *terminated = true;
                            break;
                        }
                    }
                    lane.last_collision = collision_events;
                    render_and_push(lane, preprocess, frame_stack);
                })
        });
        let mut next_states = Vec::with_capacity(count);
        let mut observations = Vec::with_capacity(count * expected);
        let mut rewards = Vec::with_capacity(count);
        let mut terminated = Vec::with_capacity(count);
        let mut signals = Vec::with_capacity(count * SIGNALS);
        for (lane, _, branch_reward, branch_terminated) in rows {
            next_states.push(serialize_lane(&lane));
            let start = observations.len();
            observations.resize(start + expected, 0);
            write_stack(&lane, &mut observations[start..], frame_stack);
            rewards.push(branch_reward);
            terminated.push(branch_terminated);
            let start = signals.len();
            signals.resize(start + SIGNALS, 0);
            write_signals(&lane, &mut signals[start..]);
        }
        Ok((next_states, observations, rewards, terminated, signals))
    }

    fn configure_lane(
        &mut self,
        lane: usize,
        paddle_x: i32,
        ball_x: i32,
        ball_y: i32,
        ball_vx: i32,
        ball_vy: i32,
        bricks: u64,
        lives: i32,
    ) -> PyResult<()> {
        let target = self
            .lanes
            .get_mut(lane)
            .ok_or_else(|| PyValueError::new_err("lane index out of range"))?;
        if lives <= 0 || bricks & !FULL_BRICKS != 0 {
            return Err(PyValueError::new_err(
                "lives must be positive and bricks must fit the 48-brick mask",
            ));
        }
        target.paddle_x = paddle_x;
        target.ball_x = ball_x;
        target.ball_y = ball_y;
        target.ball_vx = ball_vx;
        target.ball_vy = ball_vy;
        target.bricks = bricks;
        target.lives = lives;
        target.pending_reset = false;
        target.last_collision = 0;
        target.render_initialized = false;
        render_and_push(target, &self.preprocess, self.frame_stack);
        let source_slot = (target.stack_head + self.frame_stack - 1) % self.frame_stack;
        let plane = self.obs_h * self.obs_w;
        if source_slot != 0 {
            target
                .stack
                .copy_within(source_slot * plane..(source_slot + 1) * plane, 0);
        }
        for slot in 0..self.frame_stack {
            if slot > 0 {
                target.stack.copy_within(0..plane, slot * plane);
            }
        }
        target.stack_head = 0;
        Ok(())
    }
}

impl NativeBreakoutVecEnv {
    fn validate_shapes(
        &self,
        mask_len: usize,
        starts_len: usize,
        obs: &[usize],
        signals: &[usize],
    ) -> PyResult<()> {
        let expected_obs = [self.lanes.len(), self.frame_stack, self.obs_h, self.obs_w];
        if mask_len != self.lanes.len()
            || starts_len != self.lanes.len()
            || obs != expected_obs
            || signals != [self.lanes.len(), SIGNALS]
        {
            return Err(PyValueError::new_err("reset buffers have incorrect shapes"));
        }
        Ok(())
    }
    fn validate_step_shapes(
        &self,
        actions: usize,
        obs: &[usize],
        rewards: usize,
        terminated: usize,
        truncated: usize,
        signals: &[usize],
    ) -> PyResult<()> {
        let expected_obs = [self.lanes.len(), self.frame_stack, self.obs_h, self.obs_w];
        if actions != self.lanes.len()
            || obs != expected_obs
            || rewards != self.lanes.len()
            || terminated != self.lanes.len()
            || truncated != self.lanes.len()
            || signals != [self.lanes.len(), SIGNALS]
        {
            return Err(PyValueError::new_err("step buffers have incorrect shapes"));
        }
        Ok(())
    }
}

#[pymodule]
fn _breakout_turbo(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeBreakoutVecEnv>()?;
    module.add("RAW_WIDTH", RAW_W)?;
    module.add("RAW_HEIGHT", RAW_H)?;
    module.add("RENDER_WIDTH", RENDER_W)?;
    module.add("RENDER_HEIGHT", RENDER_H)?;
    module.add("FIXED_POINT_ONE", FP)?;
    Ok(())
}
