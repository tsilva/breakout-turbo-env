use numpy::{
    PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

const RAW_W: usize = 160;
const RAW_H: usize = 210;
const RENDER_W: usize = 160;
const RENDER_H: usize = 210;
const FP: i32 = 1 << 16;
const BRICK_COLS: usize = 18;
const BRICK_ROWS: usize = 6;
const BRICK_ROW_POINTS: [i32; BRICK_ROWS] = [7, 7, 4, 4, 1, 1];
const FULL_BRICKS: u128 = (1u128 << (BRICK_COLS * BRICK_ROWS)) - 1;
const SIGNALS: usize = 14;

const COLLISION_WALL: i64 = 1;
const COLLISION_PADDLE: i64 = 2;
const COLLISION_BRICK: i64 = 4;
const COLLISION_LOSS: i64 = 8;

// Stable Retro's Stella paddle is an RC circuit driven by a digital key-repeat
// emulation.  Breakout samples that circuit once per two-line kernel.  These
// are the exact lower charge boundaries of the ROM-visible measurement for
// every charge reachable through Stella's digital left/right controls.
const PADDLE_MEASURE_THRESHOLDS: [(u16, u8); 89] = [
    (1, 0),
    (272, 12),
    (295, 14),
    (318, 16),
    (341, 18),
    (366, 20),
    (388, 22),
    (411, 24),
    (447, 26),
    (470, 28),
    (493, 30),
    (516, 32),
    (539, 34),
    (563, 36),
    (586, 38),
    (609, 40),
    (633, 42),
    (657, 44),
    (680, 46),
    (703, 48),
    (733, 50),
    (757, 52),
    (780, 54),
    (803, 56),
    (828, 58),
    (851, 60),
    (874, 62),
    (897, 64),
    (920, 66),
    (943, 68),
    (966, 70),
    (991, 72),
    (1013, 74),
    (1036, 76),
    (1060, 78),
    (1083, 80),
    (1107, 82),
    (1130, 84),
    (1157, 86),
    (1180, 88),
    (1203, 90),
    (1228, 92),
    (1251, 94),
    (1274, 96),
    (1297, 98),
    (1320, 100),
    (1343, 102),
    (1366, 104),
    (1391, 106),
    (1413, 108),
    (1436, 110),
    (1460, 112),
    (1483, 114),
    (1507, 116),
    (1530, 118),
    (1553, 120),
    (1576, 122),
    (1599, 124),
    (1624, 126),
    (1647, 128),
    (1670, 130),
    (1693, 132),
    (1716, 134),
    (1739, 136),
    (1762, 138),
    (1786, 140),
    (1809, 142),
    (1832, 144),
    (1857, 146),
    (1880, 148),
    (1903, 150),
    (1926, 152),
    (1949, 154),
    (1972, 156),
    (1995, 158),
    (2020, 160),
    // The startup charge ramp reaches values unavailable once Stella's
    // repeat acceleration locks to 25; 2039 is the observed central boundary.
    (2039, 162),
    (2066, 164),
    (2088, 166),
    (2112, 168),
    (2135, 170),
    (2158, 172),
    (2182, 174),
    (2205, 176),
    (2228, 178),
    (2253, 180),
    (2276, 182),
    (2299, 184),
    (2322, 186),
];

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
    bricks: u128,
    score: i32,
    hud_score: i32,
    lives: i32,
    hud_lives: i32,
    tick: u64,
    layout_id: i32,
    pending_reset: bool,
    last_collision: i64,
    awaiting_fire: bool,
    collision_latches: u8,
    collision_count: u8,
    steep_angle: bool,
    brick_contact: bool,
    paddle_charge: u16,
    paddle_repeat: u8,
    paddle_held: bool,
    paddle_measure: u8,
    stack: Vec<u8>,
    stack_head: usize,
    raw_scratch: Vec<u8>,
}

impl Lane {
    fn new(stack_size: usize) -> Self {
        Self {
            paddle_x: 115 * FP,
            ball_x: 80 * FP,
            ball_y: 122 * FP,
            ball_vx: FP,
            ball_vy: FP,
            bricks: FULL_BRICKS,
            score: 0,
            hud_score: 0,
            lives: 5,
            hud_lives: 5,
            tick: 0,
            layout_id: 0,
            pending_reset: false,
            last_collision: 0,
            awaiting_fire: true,
            collision_latches: 0,
            collision_count: 0,
            steep_angle: true,
            brick_contact: false,
            paddle_charge: 2048,
            paddle_repeat: 0,
            paddle_held: false,
            paddle_measure: 162,
            stack: vec![0; stack_size],
            stack_head: 0,
            raw_scratch: vec![0; RAW_W * RAW_H],
        }
    }
}

fn layout_mask(layout_id: i32) -> Option<u128> {
    match layout_id {
        0 => Some(FULL_BRICKS),
        1 => {
            let mut mask = 0u128;
            for row in 0..BRICK_ROWS {
                for col in 0..BRICK_COLS {
                    if (row + col) % 2 == 0 {
                        mask |= 1u128 << (row * BRICK_COLS + col);
                    }
                }
            }
            Some(mask)
        }
        2 => {
            let mut mask = FULL_BRICKS;
            for row in 1..BRICK_ROWS {
                mask &= !(1u128 << (row * BRICK_COLS + 8));
                mask &= !(1u128 << (row * BRICK_COLS + 9));
            }
            Some(mask)
        }
        3 => {
            let mut mask = 0u128;
            for col in 0..BRICK_COLS {
                mask |= 1u128 << col;
                mask |= 1u128 << ((BRICK_ROWS - 1) * BRICK_COLS + col);
            }
            Some(mask)
        }
        _ => None,
    }
}

fn reset_lane(lane: &mut Lane, layout_id: i32, preprocess: &Preprocess, frame_stack: usize) {
    lane.paddle_x = 115 * FP;
    lane.ball_x = 80 * FP;
    lane.ball_y = 122 * FP;
    lane.ball_vx = FP;
    lane.ball_vy = FP;
    lane.bricks = layout_mask(layout_id).expect("validated layout");
    lane.score = 0;
    lane.hud_score = 0;
    lane.lives = 5;
    lane.hud_lives = 5;
    lane.tick = 0;
    lane.layout_id = layout_id;
    lane.pending_reset = false;
    lane.last_collision = 0;
    lane.awaiting_fire = true;
    lane.collision_latches = 0;
    lane.collision_count = 0;
    lane.steep_angle = true;
    lane.brick_contact = false;
    lane.paddle_charge = 2048;
    lane.paddle_repeat = 0;
    lane.paddle_held = false;
    lane.paddle_measure = 162;
    lane.stack.fill(0);
    lane.stack_head = 0;
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
    lane.hud_score = lane.score;
    let collision_paddle_x = lane.paddle_x;
    update_paddle(lane, action);
    if lane.brick_contact {
        let y = lane.ball_y / FP;
        if y > 93 || y + 3 < 56 {
            lane.brick_contact = false;
        }
    }

    if lane.awaiting_fire {
        lane.hud_lives = lane.lives;
        if action == 1 {
            let serve = ((lane.tick + 2) & 3) as usize;
            let serve_x = [16, 78, 80, 142][serve];
            lane.awaiting_fire = false;
            lane.ball_x = serve_x * FP;
            lane.ball_y = 122 * FP;
            lane.ball_vx = if serve & 1 == 0 { FP } else { -FP };
            lane.ball_vy = FP;
            lane.collision_latches = 0;
            lane.collision_count = 0;
            lane.steep_angle = true;
            lane.brick_contact = false;
        }
        lane.tick += 1;
        return (0.0, false);
    }

    if lane.ball_y / FP >= 217 {
        lane.lives -= 1;
        lane.awaiting_fire = true;
        lane.ball_y = 9 * FP;
        lane.collision_latches = 0;
        lane.last_collision |= COLLISION_LOSS;
        lane.tick += 1;
        if lane.lives <= 0 {
            lane.pending_reset = true;
            return ((lane.score - score_before) as f32, true);
        }
        return ((lane.score - score_before) as f32, false);
    }

    // The ROM consumes collision latches produced by the preceding raster
    // frame.  This one-frame delay is essential at wall/brick corners.
    if lane.collision_latches & 4 != 0 {
        let x = lane.ball_x / FP;
        if (x <= 8 && lane.ball_vx < 0) || (x >= 150 && lane.ball_vx > 0) {
            lane.ball_vx = -lane.ball_vx;
            lane.last_collision |= COLLISION_WALL;
        }
    }
    if lane.collision_latches & 1 != 0 {
        let y = lane.ball_y / FP;
        if y < 49 {
            lane.brick_contact = false;
            if lane.ball_vy < 0 {
                lane.ball_vy = -lane.ball_vy;
                lane.last_collision |= COLLISION_WALL;
            }
        } else if !lane.brick_contact {
            let index = brick_at_ball(lane);
            if visible_bricks(lane) & (1u128 << index) != 0 {
                lane.bricks &= !(1u128 << index);
                lane.score += BRICK_ROW_POINTS[index / BRICK_COLS];
                lane.ball_vy = -lane.ball_vy;
                lane.brick_contact = true;
                lane.last_collision |= COLLISION_BRICK;
            }
        }
    }
    if lane.collision_latches & 2 != 0 && lane.ball_vy > 0 {
        let relative = (collision_paddle_x + 6 * FP - lane.ball_x) / FP;
        if relative < 0 {
            lane.ball_vx = lane.ball_vx.abs();
        } else if relative > 0 {
            lane.ball_vx = -lane.ball_vx.abs();
        } else {
            lane.ball_x += if lane.ball_vx < 0 { 4 * FP } else { -4 * FP };
        }
        if relative != 0 {
            lane.steep_angle = (1..=4).contains(&relative.abs());
        }
        lane.collision_count = (lane.collision_count + 1).min(12);
        apply_atari_speed(lane);
        lane.ball_vy = -lane.ball_vy.abs();
        lane.last_collision |= COLLISION_PADDLE;
    }

    lane.ball_x += lane.ball_vx;
    lane.ball_y += lane.ball_vy;

    lane.collision_latches = raster_collision_latches(lane);
    lane.tick += 1;
    ((lane.score - score_before) as f32, false)
}

fn paddle_measurement(charge: u16) -> u8 {
    let index = PADDLE_MEASURE_THRESHOLDS.partition_point(|&(lower, _)| lower <= charge);
    PADDLE_MEASURE_THRESHOLDS[index.saturating_sub(1)].1
}

fn charge_for_paddle_measurement(measurement: u8) -> u16 {
    PADDLE_MEASURE_THRESHOLDS
        .iter()
        .min_by_key(|&&(_, value)| value.abs_diff(measurement))
        .map(|&(charge, _)| charge)
        .unwrap_or(2048)
}

fn update_paddle(lane: &mut Lane, action: u8) {
    // The ROM first smooths the prior frame's measured controller value.
    let raw_x = lane.paddle_x / FP + 47;
    let target = 235 - lane.paddle_measure as i32;
    let next_raw = ((raw_x + target) / 2).clamp(55, 191);
    lane.paddle_x = (next_raw - 47) * FP;

    if lane.paddle_held {
        lane.paddle_repeat += 1;
        if lane.paddle_repeat > 5 {
            lane.paddle_repeat = 25;
        }
    }
    match action {
        2 if lane.paddle_charge > lane.paddle_repeat as u16 => {
            lane.paddle_charge -= lane.paddle_repeat as u16;
        }
        3 if lane.paddle_charge + (lane.paddle_repeat as u16) < 3856 => {
            lane.paddle_charge += lane.paddle_repeat as u16;
        }
        _ => {}
    }
    lane.paddle_held = matches!(action, 2 | 3);
    lane.paddle_measure = paddle_measurement(lane.paddle_charge);
}

fn brick_at_ball(lane: &Lane) -> usize {
    let x = lane.ball_x / FP;
    let y = lane.ball_y / FP;
    let col = ((x - 8).max(0) / 8).min(17) as usize;
    let row = ((y - 59).max(0) / 6).min(5) as usize;
    row * BRICK_COLS + col
}

fn visible_bricks(lane: &Lane) -> u128 {
    if lane.tick > 35 {
        return lane.bricks;
    }
    let mut mask = lane.bricks;
    for phase in 0..=lane.tick as usize {
        let byte = 35 - phase;
        let row = 5 - byte % 6;
        let (first, last) = match byte / 6 {
            5 => (0, 0),
            4 => (1, 4),
            3 => (5, 8),
            2 => (9, 10),
            1 => (11, 14),
            _ => (15, 17),
        };
        for column in first..=last {
            mask &= !(1u128 << (row * BRICK_COLS + column));
        }
    }
    mask
}

fn apply_atari_speed(lane: &mut Lane) {
    let group = (lane.collision_count / 4).min(3);
    let (x, y) = match (group, lane.steep_angle) {
        (0, false) => (3 * FP / 2, FP),
        (0, true) => (FP, 3 * FP / 2),
        (1, false) => (3 * FP / 2, 2 * FP),
        (1, true) => (FP / 2, 2 * FP),
        (2, _) => (2 * FP, FP),
        _ => (2 * FP, 2 * FP),
    };
    lane.ball_vx = if lane.ball_vx < 0 { -x } else { x };
    lane.ball_vy = if lane.ball_vy < 0 { -y } else { y };
}

fn raster_collision_latches(lane: &Lane) -> u8 {
    let x = lane.ball_x / FP;
    let y = lane.ball_y / FP;
    let mut result = 0u8;
    if x <= 7 || x >= 151 {
        result |= 4;
    }
    if y <= 33 {
        result |= 1;
    } else {
        for row in 0..BRICK_ROWS {
            let top = 57 + row as i32 * 6;
            let bottom = top + 5;
            if y + 3 >= top - 1 && y <= bottom + 1 {
                let col = ((x - 8).max(0) / 8).min(17) as usize;
                if visible_bricks(lane) & (1u128 << (row * BRICK_COLS + col)) != 0 {
                    result |= 1;
                    break;
                }
            }
        }
    }
    let paddle_x = lane.paddle_x / FP;
    if y + 3 >= 189 && y <= 192 && x + 1 >= paddle_x && x <= paddle_x + 15 {
        result |= 2;
    }
    result
}

const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b010, 0b010, 0b010, 0b010],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b011, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b100, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b001, 0b001, 0b001],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b001],
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

fn render_indexed(lane: &Lane) -> Vec<u8> {
    let mut frame = vec![0u8; RENDER_W * RENDER_H];

    // Atari's status display lives inside the video signal: three score
    // digits, the selected game number, and the current player number.
    let score = lane.hud_score.clamp(0, 999) as usize;
    draw_digit(&mut frame, score / 100, 36);
    draw_digit(&mut frame, (score / 10) % 10, 52);
    draw_digit(&mut frame, score % 10, 68);
    draw_digit(&mut frame, lane.hud_lives.clamp(0, 9) as usize, 100);
    draw_digit(&mut frame, 1, 132);

    // Stella's native frame uses a 15-line header bar and 8-pixel
    // playfield walls in the same neutral gray.
    for y in 17..32 {
        frame[y * RENDER_W..(y + 1) * RENDER_W].fill(1);
    }
    for y in 32..189 {
        frame[y * RENDER_W..y * RENDER_W + 8].fill(1);
        frame[y * RENDER_W + 152..(y + 1) * RENDER_W].fill(1);
    }

    // The TIA playfield resolves the wall into 18 independently destructible
    // 8x6 cells per row.
    for row in 0..BRICK_ROWS {
        for col in 0..BRICK_COLS {
            let index = row * BRICK_COLS + col;
            if (visible_bricks(lane) >> index) & 1 == 0 {
                continue;
            }
            let x0 = 8 + col * 8;
            let y0 = 57 + row * 6;
            for y in y0..y0 + 6 {
                frame[y * RENDER_W + x0..y * RENDER_W + x0 + 8].fill(2 + row as u8);
            }
        }
    }

    let paddle_x = (lane.paddle_x / FP).clamp(8, 144) as usize;
    for y in 189..193 {
        frame[y * RENDER_W + paddle_x..y * RENDER_W + paddle_x + 16].fill(2);
    }

    if !lane.awaiting_fire {
        let ball_x = (lane.ball_x / FP).max(0) as usize;
        let ball_y = (lane.ball_y / FP).max(0) as usize;
        if ball_x < 152 && ball_y < RENDER_H {
            let visible_x0 = ball_x.max(8);
            let visible_x1 = (ball_x + 2).min(152);
            if visible_x0 < visible_x1 {
                for y in ball_y..(ball_y + 4).min(196) {
                    if (57..93).contains(&y) {
                        continue;
                    }
                    let scanline_x1 = if y == 195 {
                        visible_x1.min(47)
                    } else {
                        visible_x1
                    };
                    for x in visible_x0..scanline_x1 {
                        let pixel = &mut frame[y * RENDER_W + x];
                        if *pixel == 0 {
                            *pixel = 2;
                        }
                    }
                }
                let band_y0 = ball_y.saturating_sub(1);
                for y in band_y0..(band_y0 + 4).min(93) {
                    if y < 57 {
                        continue;
                    }
                    let ball_color = 2 + ((y - 57) / 6) as u8;
                    for x in visible_x0..visible_x1 {
                        let pixel = &mut frame[y * RENDER_W + x];
                        if *pixel == 0 {
                            *pixel = ball_color;
                        }
                    }
                }
            }
        }
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

fn render_and_push(lane: &mut Lane, preprocess: &Preprocess, frame_stack: usize) {
    let indexed = render_indexed(lane);
    for (destination, source) in lane.raw_scratch.iter_mut().zip(indexed) {
        *destination = palette_gray(source);
    }
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
    dst[13] = lane.awaiting_fire as i64;
}

fn put_i32(dst: &mut Vec<u8>, value: i32) {
    dst.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(dst: &mut Vec<u8>, value: u64) {
    dst.extend_from_slice(&value.to_le_bytes());
}
fn put_u128(dst: &mut Vec<u8>, value: u128) {
    dst.extend_from_slice(&value.to_le_bytes());
}
fn put_u16(dst: &mut Vec<u8>, value: u16) {
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
fn take_u128(src: &[u8], offset: &mut usize) -> Result<u128, &'static str> {
    let bytes: [u8; 16] = src
        .get(*offset..*offset + 16)
        .ok_or("state is truncated")?
        .try_into()
        .unwrap();
    *offset += 16;
    Ok(u128::from_le_bytes(bytes))
}
fn take_u16(src: &[u8], offset: &mut usize) -> Result<u16, &'static str> {
    let bytes: [u8; 2] = src
        .get(*offset..*offset + 2)
        .ok_or("state is truncated")?
        .try_into()
        .unwrap();
    *offset += 2;
    Ok(u16::from_le_bytes(bytes))
}

fn serialize_lane(lane: &Lane) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + lane.stack.len());
    out.extend_from_slice(b"BTO7");
    for value in [
        lane.paddle_x,
        lane.ball_x,
        lane.ball_y,
        lane.ball_vx,
        lane.ball_vy,
        lane.score,
        lane.hud_score,
        lane.lives,
        lane.hud_lives,
        lane.layout_id,
    ] {
        put_i32(&mut out, value);
    }
    put_u128(&mut out, lane.bricks);
    put_u64(&mut out, lane.tick);
    put_u64(&mut out, lane.last_collision as u64);
    put_u64(&mut out, lane.stack_head as u64);
    out.push(lane.pending_reset as u8);
    out.push(lane.awaiting_fire as u8);
    out.push(lane.collision_latches);
    out.push(lane.collision_count);
    out.push(lane.steep_angle as u8);
    out.push(lane.brick_contact as u8);
    put_u16(&mut out, lane.paddle_charge);
    out.push(lane.paddle_repeat);
    out.push(lane.paddle_held as u8);
    out.push(lane.paddle_measure);
    out.extend_from_slice(&lane.stack);
    out
}

fn deserialize_lane(data: &[u8], expected_stack: usize) -> Result<Lane, &'static str> {
    if data.get(0..4) != Some(b"BTO7") {
        return Err("state has an invalid header");
    }
    let mut offset = 4;
    let paddle_x = take_i32(data, &mut offset)?;
    let ball_x = take_i32(data, &mut offset)?;
    let ball_y = take_i32(data, &mut offset)?;
    let ball_vx = take_i32(data, &mut offset)?;
    let ball_vy = take_i32(data, &mut offset)?;
    let score = take_i32(data, &mut offset)?;
    let hud_score = take_i32(data, &mut offset)?;
    let lives = take_i32(data, &mut offset)?;
    let hud_lives = take_i32(data, &mut offset)?;
    let layout_id = take_i32(data, &mut offset)?;
    let bricks = take_u128(data, &mut offset)?;
    let tick = take_u64(data, &mut offset)?;
    let last_collision = take_u64(data, &mut offset)? as i64;
    let stack_head = take_u64(data, &mut offset)? as usize;
    let pending_reset = *data.get(offset).ok_or("state is truncated")? != 0;
    offset += 1;
    let awaiting_fire = *data.get(offset).ok_or("state is truncated")? != 0;
    offset += 1;
    let collision_latches = *data.get(offset).ok_or("state is truncated")?;
    offset += 1;
    let collision_count = *data.get(offset).ok_or("state is truncated")?;
    offset += 1;
    let steep_angle = *data.get(offset).ok_or("state is truncated")? != 0;
    offset += 1;
    let brick_contact = *data.get(offset).ok_or("state is truncated")? != 0;
    offset += 1;
    let paddle_charge = take_u16(data, &mut offset)?;
    let paddle_repeat = *data.get(offset).ok_or("state is truncated")?;
    offset += 1;
    let paddle_held = *data.get(offset).ok_or("state is truncated")? != 0;
    offset += 1;
    let paddle_measure = *data.get(offset).ok_or("state is truncated")?;
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
        hud_score,
        lives,
        hud_lives,
        tick,
        layout_id,
        pending_reset,
        last_collision,
        awaiting_fire,
        collision_latches,
        collision_count,
        steep_angle,
        brick_contact,
        paddle_charge,
        paddle_repeat,
        paddle_held,
        paddle_measure,
        stack,
        stack_head,
        raw_scratch: vec![0; RAW_W * RAW_H],
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
        if let Some(action) = actions.iter().find(|&&action| action > 3) {
            return Err(PyValueError::new_err(format!(
                "actions must be 0, 1, 2, or 3; got {action}"
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
        if actions.iter().any(|&action| action > 3) {
            return Err(PyValueError::new_err("actions must be 0, 1, 2, or 3"));
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
        bricks: u128,
        lives: i32,
    ) -> PyResult<()> {
        let target = self
            .lanes
            .get_mut(lane)
            .ok_or_else(|| PyValueError::new_err("lane index out of range"))?;
        if lives <= 0 {
            return Err(PyValueError::new_err("lives must be positive"));
        }
        target.paddle_x = paddle_x;
        let raw_paddle_x = (paddle_x / FP + 47).clamp(55, 191);
        target.paddle_measure = (235 - raw_paddle_x) as u8;
        target.paddle_charge = charge_for_paddle_measurement(target.paddle_measure);
        target.paddle_repeat = 0;
        target.paddle_held = false;
        target.ball_x = ball_x;
        target.ball_y = ball_y;
        target.ball_vx = ball_vx;
        target.ball_vy = ball_vy;
        target.bricks = bricks;
        target.tick = 36;
        target.lives = lives;
        target.pending_reset = false;
        target.last_collision = 0;
        target.awaiting_fire = false;
        target.collision_latches = 0;
        target.brick_contact = false;
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
