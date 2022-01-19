use std::{env, time::Instant};

use minifb::{Key, Scale, Window, WindowOptions};
use sim::MAX_ALT;
use vek::{Vec2, Vec3};

use crate::sim::WeatherSim;

mod grid;
mod sim;

const WIDTH: usize = 256;
const HEIGHT: usize = 256;

enum DisplayMode {
    Altitude,
    Clouds,
    Wind,
}

impl DisplayMode {
    fn display(&self, buffer: &mut Vec<f32>, sim: &WeatherSim) {
        let weather = sim.get_weather();
        let consts = sim.get_consts();
        match self {
            DisplayMode::Altitude => {
                for (p, c) in consts.iter() {
                    let t = c.alt / MAX_ALT;
                    let i = (p.y as usize * WIDTH + p.x as usize) * 3;
                    buffer[i] = t;
                    buffer[i + 1] = t;
                    buffer[i + 2] = t;
                }
            }
            DisplayMode::Clouds => {
                for (p, weather) in weather.iter() {
                    let t = weather.cloud / 3.0;
                    let i = (p.y as usize * WIDTH + p.x as usize) * 3;
                    buffer[i] += t;
                    buffer[i + 1] += t;
                    buffer[i + 2] += t;
                }
            }
            DisplayMode::Wind => {
                for (p, weather) in weather.iter() {
                    let i = (p.y as usize * WIDTH + p.x as usize) * 3;
                    let wind = if weather.wind.magnitude_squared() > f32::EPSILON {
                        (weather.wind.normalized() + 1.0) / 2.0 / 3.0
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
        match args[1].as_str() {
            "alt" => DisplayMode::Altitude,
            "cloud" => DisplayMode::Clouds,
            "wind" => DisplayMode::Wind,
            _ => panic!("Invalid display mode"),
        }
    } else {
        DisplayMode::Clouds
    };

    let mut float_buffer: Vec<f32> = vec![0.0; WIDTH * HEIGHT * 3];
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
        if (now - last_update).as_secs_f64() > 0.01 {
            sim.tick((now - first_update).as_secs_f64());
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
