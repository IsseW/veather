use std::{env, ops::Range, time::Instant};

use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use sim::MAX_ALT;
use vek::{Vec2, Vec3};

use crate::sim::WeatherSim;

mod grid;
mod sim;

const WIDTH: usize = 250;
const HEIGHT: usize = 250;

const WANTED_WIDTH: usize = 600;
const WANTED_HEIGHT: usize = 600;

const WINDOW_WIDTH: usize = (WANTED_WIDTH / WIDTH) * WIDTH;
const WINDOW_HEIGHT: usize = (WANTED_HEIGHT / HEIGHT) * HEIGHT;

const CELL_WIDTH: usize = WINDOW_WIDTH / WIDTH;
const CELL_HEIGHT: usize = WINDOW_HEIGHT / HEIGHT;

enum DisplayKind {
    Altitude,
    Clouds,
    Wind,
    Rain,
}

struct DisplayMode {
    kind: DisplayKind,
    range: Range<i32>,
}

impl DisplayMode {
    fn display(&self, buffer: &mut Vec<f64>, sim: &WeatherSim) {
        let weather = sim.get_weather();
        let consts = sim.get_consts();
        match self.kind {
            DisplayKind::Altitude => {
                for (p, c) in consts.iter() {
                    let t = c.alt / MAX_ALT;
                    let i = (p.y as usize * WIDTH + p.x as usize) * 3;
                    buffer[i] = t;
                    buffer[i + 1] = t;
                    buffer[i + 2] = t;
                }
            }
            DisplayKind::Clouds => {
                for (p, weather) in weather.iter() {
                    if !self.range.contains(&p.z) {
                        continue;
                    }
                    let t = weather.cloud / self.range.len() as f64;
                    let i = (p.y as usize * WIDTH + p.x as usize) * 3;
                    buffer[i] += t;
                    buffer[i + 1] += t;
                    buffer[i + 2] += t;
                }
            }
            DisplayKind::Rain => {
                for (p, weather) in weather.iter() {
                    if !self.range.contains(&p.z) {
                        continue;
                    }
                    let t = weather.rain / self.range.len() as f64;
                    let i = (p.y as usize * WIDTH + p.x as usize) * 3;
                    buffer[i] += t;
                    buffer[i + 1] += t;
                    buffer[i + 2] += t;
                }
            }
            DisplayKind::Wind => {
                for (p, weather) in weather.iter() {
                    if !self.range.contains(&p.z) {
                        continue;
                    }
                    let i = (p.y as usize * WIDTH + p.x as usize) * 3;
                    let wind = if weather.wind.magnitude_squared() > f64::EPSILON {
                        (weather.wind.normalized() + 1.0) / 2.0 / self.range.len() as f64
                    } else {
                        Vec3::zero()
                    };

                    buffer[i] += wind.x;
                    buffer[i + 1] += wind.y;
                    buffer[i + 2] += wind.z;
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut display_mode = if args.len() > 1 {
        let range = args
            .get(2)
            .map(|s| {
                s.split_once("..")
                    .map(|(a, b)| {
                        a.parse::<i32>()
                            .ok()
                            .zip(b.parse::<i32>().ok())
                            .map(|(a, b)| a..b)
                    })
                    .flatten()
            })
            .flatten()
            .unwrap_or(0..3);
        match args[1].as_str() {
            "alt" => DisplayMode {
                kind: DisplayKind::Altitude,
                range,
            },
            "cloud" => DisplayMode {
                kind: DisplayKind::Clouds,
                range,
            },
            "wind" => DisplayMode {
                kind: DisplayKind::Wind,
                range,
            },
            "rain" => DisplayMode {
                kind: DisplayKind::Rain,
                range,
            },
            _ => panic!("Invalid display mode"),
        }
    } else {
        DisplayMode {
            kind: DisplayKind::Clouds,
            range: 0..3,
        }
    };

    let mut float_buffer: Vec<f64> = vec![0.0; WIDTH * HEIGHT * 3];
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut display_buffer: Vec<u32> = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT];

    let options = WindowOptions {
        scale: Scale::X2,
        ..WindowOptions::default()
    };
    let mut window = Window::new("ESC to exit", WINDOW_WIDTH, WINDOW_HEIGHT, options)
        .expect("Unable to open window");

    let mut sim = WeatherSim::new(Vec2::new(WIDTH, HEIGHT).as_());
    let first_update = Instant::now();
    let mut last_update = first_update;
    let mut tick: u64 = 0;
    let mut auto_play = false;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let (redraw, do_tick) = {
            if window.is_key_pressed(Key::Space, KeyRepeat::No) {
                auto_play = !auto_play;
            }
            if window.is_key_pressed(Key::C, KeyRepeat::No) {
                display_mode.kind = DisplayKind::Clouds;
                (true, false)
            } else if window.is_key_pressed(Key::W, KeyRepeat::No) {
                display_mode.kind = DisplayKind::Wind;
                (true, false)
            } else if auto_play || window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
                (true, true)
            } else {
                (false, false)
            }
        };
        if do_tick {
            sim.tick(1.0);
        }

        if redraw {
            float_buffer.iter_mut().for_each(|x| *x = 0.0);

            display_mode.display(&mut float_buffer, &sim);
            for i in 0..WIDTH * HEIGHT {
                let a = 255.0 * (float_buffer[i * 3]).min(1.0);
                let b = 255.0 * (float_buffer[i * 3 + 1]).min(1.0);
                let c = 255.0 * (float_buffer[i * 3 + 2]).min(1.0);
                let color = (a as u32) << 16 | (b as u32) << 8 | c as u32;
                buffer[i] = color;
            }
            for y in 0..WINDOW_HEIGHT {
                for x in 0..WINDOW_WIDTH {
                    display_buffer[y * WINDOW_WIDTH + x] =
                        buffer[y / CELL_HEIGHT * WIDTH + x / CELL_WIDTH];
                }
            }
            last_update = now;
            tick += 1;
            window.set_title(format!("{}", tick).as_str());
        }

        window
            .update_with_buffer(&display_buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
