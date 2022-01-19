use crate::grid::{Grid, Volume};
use lazy_static::lazy_static;
use simdnoise::NoiseBuilder;
use vek::*;

#[derive(Default, Clone)]
pub struct Constants {
    pub alt: f32,
    pub normal: Vec3<f32>,
    pub humidity: f32,
    pub temp: f32,
}

#[derive(Default, Clone)]
pub struct Cell {
    pub wind: Vec3<f32>,
    pub temperature: f32,
    pub moisture: f32,
    pub cloud: f32,
}

impl std::ops::Mul<f32> for Cell {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self {
            wind: self.wind * rhs,
            temperature: self.temperature * rhs,
            moisture: self.moisture * rhs,
            cloud: self.cloud * rhs,
        }
    }
}

impl std::ops::Add<Cell> for Cell {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            wind: self.wind + rhs.wind,
            temperature: self.temperature + rhs.temperature,
            moisture: self.moisture + rhs.moisture,
            cloud: self.cloud + rhs.cloud,
        }
    }
}

lazy_static! {
    static ref NOISE: Volume<Cell> = {
        const SIZE: usize = 1024;
        let mut volume = Volume::new(Vec3::new(SIZE, SIZE, 3).as_(), Cell::default());

        let wind_x = NoiseBuilder::fbm_3d(SIZE, SIZE, 3)
            .with_seed(69)
            .generate_scaled(-1.0, 1.0);
        let wind_y = NoiseBuilder::fbm_3d(SIZE, SIZE, 3)
            .with_seed(169)
            .generate_scaled(-40.0, 40.0);

        let temperature = NoiseBuilder::fbm_3d(SIZE, SIZE, 3)
            .with_seed(269)
            .generate_scaled(0.5, 1.0);

        let moisture = NoiseBuilder::fbm_3d(SIZE, SIZE, 3)
            .with_seed(369)
            .generate_scaled(0.0, 0.4);

        let cloud = NoiseBuilder::fbm_3d(SIZE, SIZE, 3)
            .with_seed(469)
            .generate_scaled(0.0, 0.4);

        volume.iter_mut().enumerate().for_each(|(i, (pos, cell))| {
            cell.wind =
                Vec3::new(wind_x[i].powi(3), wind_y[i].powi(3), 0.0) * 30.0 / (3 - pos.z) as f32;
            cell.temperature = temperature[i].powi(3) * 30.0;
            cell.moisture = moisture[i].powi(3);
            cell.cloud = cloud[i].powi(3);
        });

        volume
    };
}

/// Used to sample weather that isn't simulated
fn sample_cell(p: Vec3<i32>, time: f64) -> Cell {
    // return Cell::default();
    let p0 = NOISE
        .get((p + NOISE.size()) % NOISE.size())
        .unwrap()
        .clone();
    let p1 = NOISE
        .get((p + Vec3::new(1, 1, 0) + NOISE.size()) % NOISE.size())
        .unwrap()
        .clone();
    let t = ((time / 24.0) % 1.0) as f32;
    p0 * (1.0 - t) + p1 * t
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Weather {
    /// Clouds currently in the area between 0 and 1
    pub cloud: f32,
    /// Rain per time, between 0 and 1
    pub rain: f32,
    // Wind direction in block / second
    pub wind: Vec3<f32>,
}

#[derive(Clone, Copy, Default)]
pub struct WeatherInfo {
    pub lightning_chance: f32,
}

pub struct WeatherSim {
    cells: Volume<Cell>,      // The variables used for simulation
    weather: Volume<Weather>, // The current weather.
    consts: Grid<Constants>,  // The constants from the world used for simulation
    info: Grid<WeatherInfo>,
}

const WATER_BOILING_POINT: f32 = 100.0;
const MAX_WIND_SPEED: f32 = 128.0;
pub(crate) const CELL_SIZE: f32 = (16 * 32) as f32;
pub(crate) const DT: f32 = CELL_SIZE / MAX_WIND_SPEED;
pub(crate) const MAX_ALT: f32 = 2000.0;
pub(crate) const SIM_ALT: f32 = 4500.0;
pub(crate) const CELL_HEIGHT: f32 = SIM_ALT / 3.0;

fn get_normal(p: f32, left: f32, right: f32, bottom: f32, top: f32) -> Vec3<f32> {
    let mut x0 = Vec3::new(CELL_SIZE, 0.0, p - left)
        .cross(Vec3::new(0.0, CELL_SIZE, 0.0))
        .normalized();
    if x0.z < 0.0 {
        x0 = -x0;
    }
    let mut x1 = Vec3::new(-CELL_SIZE, 0.0, p - right)
        .cross(Vec3::new(0.0, CELL_SIZE, 0.0))
        .normalized();
    if x1.z < 0.0 {
        x1 = -x1;
    }
    let mut y0 = Vec3::new(0.0, CELL_SIZE, p - bottom)
        .cross(Vec3::new(CELL_SIZE, 0.0, 0.0))
        .normalized();
    if y0.z < 0.0 {
        y0 = -y0;
    }
    let mut y1 = Vec3::new(0.0, -CELL_SIZE, p - top)
        .cross(Vec3::new(CELL_SIZE, 0.0, 0.0))
        .normalized();
    if y1.z < 0.0 {
        y1 = -y1;
    }
    ((x0 + x1 + y0 + y1) / 4.0).normalized()
}

pub fn iter_index(start: Vec3<i32>, size: Vec3<i32>) -> impl Iterator<Item = Vec3<i32>> {
    (start.z..size.z)
        .map(move |z| {
            (start.y..size.y)
                .map(move |y| (start.x..size.x).map(move |x| Vec3::new(x, y, z)))
                .flatten()
        })
        .flatten()
}

impl WeatherSim {
    pub fn new(size: Vec2<u32>) -> Self {
        let size = size.as_();
        let mut this = Self {
            cells: Volume::new(size.with_z(3), Cell::default()),
            weather: Volume::new(size.with_z(3), Weather::default()),
            consts: Grid::from_raw(size, {
                let size: Vec2<usize> = size.as_();
                let alts = NoiseBuilder::fbm_2d(size.x as usize, size.y as usize)
                    .with_seed(69)
                    .generate_scaled(0.0, 1.0);
                let humidities = NoiseBuilder::fbm_2d(size.x as usize, size.y as usize)
                    .with_seed(420)
                    .generate_scaled(0.0, 1.0);
                let temperatures = NoiseBuilder::fbm_2d(size.x as usize, size.y as usize)
                    .with_seed(1337)
                    .generate_scaled(-30.0, 30.0);
                let mut consts = (0..size.x * size.y)
                    .map(|i| {
                        let p = Vec2::new(i % size.x, i / size.x);
                        let extent = size.x.min(size.y) as f32 / 2.0;
                        Constants {
                            alt: (alts[i].powi(3) * MAX_ALT + 500.0)
                                * (1.0
                                    - p.as_::<f32>()
                                        .distance(Vec2::new(size.x / 2, size.y / 2).as_())
                                        / (extent)),
                            humidity: humidities[i],
                            temp: temperatures[i],
                            normal: Vec3::zero(),
                        }
                    })
                    .collect::<Vec<_>>();
                for i in 0..size.x * size.y {
                    consts[i].normal = get_normal(
                        consts[i].alt,
                        alts[(i.checked_sub(1).unwrap_or(0)).max((i / size.x) * size.x)],
                        alts[(i + 1).min((i / size.x + 1) * size.x - 1)],
                        alts[if (i as isize - size.x as isize) < 0 {
                            i
                        } else {
                            i - size.x
                        }],
                        alts[if i + size.x >= size.x * size.y {
                            i
                        } else {
                            i + size.x
                        }],
                    );
                }
                consts
            }),
            info: Grid::new(size, WeatherInfo::default()),
        };
        this.cells.iter_mut().for_each(|(point, cell)| {
            let time = 0.0;
            *cell = sample_cell(point, time);
        });
        this
    }

    pub fn get_weather(&self) -> &Volume<Weather> {
        &self.weather
    }

    pub fn get_cell(&self, p: Vec3<i32>, time: f64) -> Cell {
        self.cells
            .get(p)
            .cloned()
            .unwrap_or_else(|| sample_cell(p, time))
    }

    pub fn get_consts(&self) -> &Grid<Constants> {
        &self.consts
    }

    // https://minds.wisconsin.edu/bitstream/handle/1793/66950/LitzauSpr2013.pdf
    // https://miis.maths.ox.ac.uk/512/1/App-of-Cellular-automata-to-weather-radar.pdf
    // Time step is cell size / maximum wind speed
    pub fn tick(&mut self, time: f64) {
        // Can be calculated at start instead.
        let points_outer: Vec<_> = iter_index(
            Vec3::new(-1, -1, 0),
            (self.cells.size().xy() + 2).with_z(self.cells.size().z),
        )
        .filter(|p| {
            self.consts
                .get(p.xy())
                .map_or(true, |c| c.alt < CELL_HEIGHT * (p.z + 1) as f32)
        })
        .collect();
        let points: Vec<_> = self
            .cells
            .iter_index()
            .filter(|p| self.consts[p.xy()].alt < CELL_HEIGHT * (p.z + 1) as f32)
            .collect();

        let mut swap = Volume::new(self.cells.size(), Cell::default());

        // Disperse
        for &point in &points_outer {
            fn get_spread_volume(dir: Vec3<i32>) -> f32 {
                const HORIZONTAL_DISSIPATION: f32 = 0.1;
                const VERTICAL_DISSIPATION: f32 = 0.02;
                let spread_x = CELL_SIZE * HORIZONTAL_DISSIPATION;
                let spread_y = CELL_SIZE * VERTICAL_DISSIPATION;
                (if dir.x == 0 { CELL_SIZE } else { spread_x })
                    * (if dir.y == 0 { CELL_SIZE } else { spread_x })
                    * (if dir.z == 0 { CELL_SIZE } else { spread_y })
            }

            let neighbors: Vec<_> = (0..27)
                .map(|i| Vec3::new(1 - i / 9, 1 - i % 9 / 3, 1 - i % 3))
                .filter(|p| {
                    let p = p + point;

                    (0..self.cells.size().x).contains(&p.x)
                        && (0..self.cells.size().y).contains(&p.y)
                        && (0..self.cells.size().z).contains(&p.z)
                        // Don't interact with cells that are underground
                        && self.consts[p.xy()].alt < CELL_HEIGHT * (p.z + 1) as f32
                })
                .map(|dir| (dir, get_spread_volume(dir)))
                .collect();

            let spread_volume: f32 = neighbors.iter().map(|(_, vol)| vol).sum();
            let cell = self.get_cell(point, time);

            for (neighbor, vol) in neighbors {
                let p = neighbor + point;
                let part = vol / spread_volume;
                swap[p].cloud = fractional_add(swap[p].cloud, cell.cloud * part, 2.0);
                swap[p].moisture = fractional_add(swap[p].moisture, cell.moisture * part, 2.5);
                swap[p].temperature += cell.temperature * part;
                swap[p].wind += cell.wind * part;
            }
        }

        // Wind
        for &point in &points_outer {
            let cell = self.get_cell(point, time);
            let dir_vec = {
                let mut dir = cell.wind;
                dir.apply(|e| e.signum());
                dir.as_::<i32>()
            };

            let get_spread_volume = |dir: Vec3<i32>, vel: Vec3<f32>| {
                (if dir.x == 0 { CELL_SIZE - vel.x } else { vel.x })
                    * (if dir.y == 0 { CELL_SIZE - vel.y } else { vel.y })
                    * (if dir.z == 0 { CELL_SIZE - vel.z } else { vel.z })
            };

            let neighbors: Vec<_> = (0..8)
                .map(|i| dir_vec * Vec3::new(i / 4, i / 2 % 2, i % 2))
                .filter(|&p| {
                    let p = p + point;

                    (0..self.cells.size().x).contains(&p.x)
                        && (0..self.cells.size().y).contains(&p.y)
                        && (0..self.cells.size().z).contains(&p.z)
                        // Don't interact with cells that are underground
                        && self.consts[p.xy()].alt < CELL_HEIGHT * (p.z + 1) as f32
                })
                .map(|dir| (dir, get_spread_volume(dir, cell.wind)))
                .collect();
            let spread_volume: f32 = neighbors.iter().map(|(_, vol)| vol).sum();

            let cell = swap
                .get(point)
                .cloned()
                .unwrap_or_else(|| sample_cell(point, time));
            for (neighbor, vol) in neighbors {
                let p = neighbor + point;
                if swap.get(p).is_some() {
                    let part = vol / spread_volume;
                    self.cells[p].cloud = fractional_add(swap[p].cloud, cell.cloud * part, 2.0);
                    self.cells[p].moisture =
                        fractional_add(swap[p].moisture, cell.moisture * part, 2.5);
                    self.cells[p].temperature += cell.temperature * part;
                    self.cells[p].wind += cell.wind * part;
                }
            }
        }
        let mut max_temp = 0.0;
        let mut max_wind = Vec3::zero();
        let mut max_moisture = 0.0;
        let mut max_cloud = 0.0;
        let mut max_condens = 0.0;
        for &point in &points {
            // Some variables only apply if the ground is within the cell.
            let grounded = point.z as f32 * CELL_HEIGHT < self.consts[point.xy()].alt;

            self.cells[point].temperature = if grounded {
                f32::lerp(swap[point].temperature, self.consts[point.xy()].temp, 0.1)
            } else {
                swap[point].temperature
            };

            // Deflect and apply friction to wind.
            self.cells[point].wind = if grounded {
                let reflect = if swap[point].wind.dot(self.consts[point.xy()].normal) < 0.0 {
                    swap[point].wind.reflected(self.consts[point.xy()].normal)
                } else {
                    swap[point].wind
                };
                let friction =
                    (self.consts[point.xy()].alt - point.z as f32 * CELL_HEIGHT) / CELL_HEIGHT;

                Vec3::lerp(swap[point].wind, reflect, friction * 0.7) * (1.0 - friction * 0.01)
            } else {
                swap[point].wind
            };

            // Constants NOAA use. https://en.wikipedia.org/wiki/National_Oceanic_and_Atmospheric_Administration
            // There are other sets of constants, might be worth to give them a try
            const B: f32 = 18.678;
            const C: f32 = 257.14;
            // he dew point is the temperature to which air must be cooled to become saturated with water vapor https://en.wikipedia.org/wiki/Dew_point
            let dew_point = (swap[point].moisture / 100.0).ln()
                + B * self.cells[point].temperature / (C + self.cells[point].temperature);
            // TODO: convert into a function
            const CRITICAL_UPDRAUGHT: f32 = 20.0;
            let evaporation = if grounded && self.cells[point].temperature >= CRITICAL_UPDRAUGHT {
                self.consts[point.xy()].humidity
                    * (self.cells[point].temperature - CRITICAL_UPDRAUGHT)
            } else {
                0.0
            };

            let condensation = if dew_point >= self.cells[point].temperature {
                // Moisture -> Cloud
                0.25 * swap[point].moisture
            } else {
                // Cloud -> Moisture
                -0.5 * swap[point].cloud
            };

            const LATENT_MOISTURE_HEAT: f32 = 0.01;
            // Temperature change arising from condensation
            self.cells[point].temperature -= LATENT_MOISTURE_HEAT * condensation;

            self.cells[point].moisture = fractional_add(
                if grounded {
                    f32::lerp(swap[point].moisture, self.consts[point.xy()].humidity, 0.1)
                } else {
                    swap[point].moisture
                },
                condensation + evaporation,
                2.5,
            );

            const RAIN_AMOUNT: f32 = 0.02;
            // At what cloud density it starts raining
            // TODO: Make this a function of temperature?
            const RAIN_CRITICAL: f32 = 0.2;
            let rain = if swap[point].cloud > RAIN_CRITICAL {
                RAIN_AMOUNT * (swap[point].cloud - RAIN_CRITICAL)
            } else {
                0.0
            };
            self.cells[point].cloud = fractional_add(swap[point].cloud, condensation - rain, 2.0);

            self.weather[point].wind = self.cells[point].wind;
            self.weather[point].cloud = self.cells[point].cloud * 100.0;
            self.weather[point].rain = rain;

            if self.cells[point].temperature > max_temp {
                max_temp = self.cells[point].temperature;
            }
            if self.cells[point].wind.magnitude() > max_wind.magnitude() {
                max_wind = self.cells[point].wind;
            }
            if self.cells[point].moisture > max_moisture {
                max_moisture = self.cells[point].moisture;
            }
            if self.cells[point].cloud > max_cloud {
                max_cloud = self.cells[point].cloud;
            }
            if condensation < self.consts[point.xy()].temp {
                max_condens = condensation;
            }
        }
        println!("Maxes:\n\tTemperature: {max_temp}\n\tWind: {max_wind}\n\tMoisture: {max_moisture}\n\tCloud: {max_cloud}\n\tCondensation: {max_condens}");
    }
}

/// Adds an arbitrary float to a number between 0 and 1. And keeps the result between 0 and 1.
fn fractional_add(a: f32, b: f32, scale: f32) -> f32 {
    if b >= 0.0 {
        a + (1.0 - a) * b / (scale + b)
    } else {
        a - a * b / (scale + b)
    }
}
