use std::{env, ops::Range, time::Instant};

use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use sim::MAX_ALT;
use vek::{Vec2, Vec3};

use crate::sim::WeatherSim;

mod grid;
mod sim;

const WIDTH: usize = 256;
const HEIGHT: usize = 256;

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

    let display_mode = if args.len() > 1 {
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
    let mut buffer: Vec<u32> = vec![0x00_FF_FF_FF; WIDTH * HEIGHT];

    let options = WindowOptions {
        scale: Scale::X2,
        ..WindowOptions::default()
    };
    let mut window =
        Window::new("ESC to exit", WIDTH, HEIGHT, options).expect("Unable to open window");

    let mut sim = WeatherSim::new(Vec2::new(WIDTH, HEIGHT).as_());
    let first_update = Instant::now();
    let mut last_update = first_update;
    let mut tick: u64 = 0;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        if window.is_key_down(Key::Space) || window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
            sim.tick(1.0);
            float_buffer.iter_mut().for_each(|x| *x = 0.0);

            display_mode.display(&mut float_buffer, &sim);
            for i in 0..WIDTH * HEIGHT {
                let a = 255.0 * (float_buffer[i * 3]).min(1.0);
                let b = 255.0 * (float_buffer[i * 3 + 1]).min(1.0);
                let c = 255.0 * (float_buffer[i * 3 + 2]).min(1.0);
                let color = (a as u32) << 16 | (b as u32) << 8 | c as u32;
                buffer[i] = color;
            }
            last_update = now;
            tick += 1;
            window.set_title(format!("{}", tick).as_str());
        }
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
    }
}
