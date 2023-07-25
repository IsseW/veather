#![feature(let_chains)]
use std::{env, ops::Range, time::Instant};

use show_image::{create_window, event::{WindowEvent, VirtualKeyCode}, ImageView, ImageInfo};
use sim::MAX_ALT;
use vek::{Vec2, Vec3};

use crate::sim::WeatherSim;

mod grid;
mod sim;

const WIDTH: usize = 250;
const HEIGHT: usize = 250;

const TARGET_WIDTH: usize = 600;
const TARGET_HEIGHT: usize = 600;

const WINDOW_WIDTH: usize = (TARGET_WIDTH / WIDTH) * WIDTH;
const WINDOW_HEIGHT: usize = (TARGET_HEIGHT / HEIGHT) * HEIGHT;

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


#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let mut display_buffer: Vec<u8> = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT * 3];

    let window = create_window("veather", Default::default())?;

    let mut draw = |sim: &WeatherSim, display_mode: &DisplayMode, tick| {
        float_buffer.iter_mut().for_each(|x| *x = 0.0);

        display_mode.display(&mut float_buffer, &sim);
        for i in 0..WIDTH * HEIGHT {
            let r = (255.0 * (float_buffer[i * 3]).min(1.0)) as u8;
            let b = (255.0 * (float_buffer[i * 3 + 1]).min(1.0)) as u8;
            let g = (255.0 * (float_buffer[i * 3 + 2]).min(1.0)) as u8;
            
            let y = CELL_HEIGHT * (i / WIDTH);
            let x = CELL_WIDTH * (i % WIDTH);
            for y in y..y + CELL_HEIGHT {
                for x in x..x + CELL_WIDTH {
                    let i = y * WINDOW_WIDTH + x;
                    display_buffer[i * 3] = r;
                    display_buffer[i * 3 + 1] = b;
                    display_buffer[i * 3 + 2] = g;
                }
            }
        }
        let image = ImageView::new(ImageInfo::rgb8(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32), &display_buffer);
        window.set_image(format!("veather: {}", tick).as_str(), image)
    };

    let mut sim = WeatherSim::new(Vec2::new(WIDTH, HEIGHT).as_());
    let mut tick: u64 = 0;

    draw(&sim, &display_mode, tick)?;

    for event in window.event_channel()? {
        if let WindowEvent::KeyboardInput(event) = event && event.input.state.is_pressed() && let Some(key_code) = event.input.key_code {
            let redraw = match key_code {
                VirtualKeyCode::Escape => break,
                VirtualKeyCode::C => {
                    display_mode.kind = DisplayKind::Clouds;
                    true
                },
                VirtualKeyCode::W => {
                    display_mode.kind = DisplayKind::Wind;
                    true
                },
                VirtualKeyCode::R => {
                    display_mode.kind = DisplayKind::Rain;
                    true
                },
                VirtualKeyCode::A => {
                    display_mode.kind = DisplayKind::Altitude;
                    true
                },
                VirtualKeyCode::Right => {
                    sim.tick(tick as f64 * sim::DT);
                    tick += 1;
                    true
                },

                _ => false
            };

            if redraw {
                draw(&sim, &display_mode, tick)?;
            }
        }
    }
    Ok(())
}
