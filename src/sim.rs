use crate::grid::{Grid, Volume};
use lazy_static::lazy_static;
use ordered_float::NotNan;
use simdnoise::NoiseBuilder;
use vek::*;

#[derive(Default, Clone)]
pub struct Constants {
    pub alt: f64,
    pub normal: Vec3<f64>,
    pub humidity: f64,
    pub temp: f64,
}

#[derive(Default, Clone)]
pub struct Cell {
    pub wind: Vec3<f64>,
    pub temperature: f64,
    pub moisture: f64,
    pub cloud: f64,
}

impl std::ops::Mul<f64> for Cell {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
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
            cell.wind = Vec3::new((wind_x[i] as f64).powi(3), (wind_y[i] as f64).powi(3), 0.0)
                * 30.0
                / (3 - pos.z) as f64;
            cell.temperature = (temperature[i] as f64).powi(3) * 30.0;
            cell.moisture = (moisture[i] as f64).powi(3);
            cell.cloud = (cloud[i] as f64).powi(3);
        });

        volume
    };
}

/// Used to sample weather that isn't simulated
fn sample_cell(p: Vec3<i32>, time: f64) -> Cell {
    // return Cell::default();
    //let p0 = NOISE
    //    .get((p + NOISE.size()) % NOISE.size())
    //    .unwrap()
    //    .clone();
    //let p1 = NOISE
    //    .get((p + Vec3::new(1, 1, 0) + NOISE.size()) % NOISE.size())
    //    .unwrap()
    //    .clone();
    //let t = ((time / 24.0) % 1.0) as f64;
    //p0 * (1.0 - t) + p1 * t
    Cell {
        wind: Vec3::new(10.0, 10.0, 0.0),
        temperature: 10.0,
        moisture: 0.5,
        cloud: if p.x % 8 == p.y % 8 { 1.0 } else { 0.0 },
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Weather {
    /// Clouds currently in the area between 0 and 1
    pub cloud: f64,
    /// Rain per time, between 0 and 1
    pub rain: f64,
    // Wind direction in block / second
    pub wind: Vec3<f64>,
}

#[derive(Clone, Copy, Default)]
pub struct WeatherInfo {
    pub lightning_chance: f64,
}

pub struct WeatherSim {
    cells: Volume<Cell>,      // The variables used for simulation
    weather: Volume<Weather>, // The current weather.
    consts: Grid<Constants>,  // The constants from the world used for simulation
    info: Grid<WeatherInfo>,
}

const WATER_BOILING_POINT: f64 = 100.0;
const MAX_WIND_SPEED: f64 = 128.0;
pub(crate) const CELL_SIZE: f64 = (16 * 32) as f64;
pub(crate) const DT: f64 = CELL_SIZE / MAX_WIND_SPEED;
pub(crate) const MAX_ALT: f64 = 2000.0;
pub(crate) const SIM_ALT: f64 = 4500.0;
pub(crate) const CELL_HEIGHT: f64 = SIM_ALT / 3.0;

fn get_normal(p: f64, left: f64, right: f64, bottom: f64, top: f64) -> Vec3<f64> {
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
                        let extent = size.x.min(size.y) as f64 / 2.0;
                        Constants {
                            alt: ((alts[i] as f64).powi(3) * MAX_ALT + 500.0)
                                * (1.0
                                    - p.as_::<f64>()
                                        .distance(Vec2::new(size.x / 2, size.y / 2).as_())
                                        / (extent)),
                            humidity: humidities[i] as f64,
                            temp: temperatures[i] as f64,
                            normal: Vec3::zero(),
                        }
                    })
                    .collect::<Vec<_>>();
                for i in 0..size.x * size.y {
                    consts[i].normal = get_normal(
                        consts[i].alt,
                        consts[(i.checked_sub(1).unwrap_or(0)).max((i / size.x) * size.x)].alt,
                        consts[(i + 1).min((i / size.x + 1) * size.x - 1)].alt,
                        consts[if (i as isize - size.x as isize) < 0 {
                            i
                        } else {
                            i - size.x
                        }]
                        .alt,
                        consts[if i + size.x >= size.x * size.y {
                            i
                        } else {
                            i + size.x
                        }]
                        .alt,
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
                .map_or(true, |c| c.alt < CELL_HEIGHT * (p.z + 1) as f64)
        })
        .collect();
        let points: Vec<_> = self
            .cells
            .iter_index()
            .filter(|p| self.consts[p.xy()].alt < CELL_HEIGHT * (p.z + 1) as f64)
            .collect();

        let mut swap = Volume::new(self.cells.size(), Cell::default());

        // Disperse
        for &point in &points_outer {
            fn get_spread_volume(dir: Vec3<i32>) -> f64 {
                const HORIZONTAL_DISSIPATION: f64 = 0.01;
                const VERTICAL_DISSIPATION: f64 = 0.004;
                let spread_x = CELL_SIZE * HORIZONTAL_DISSIPATION;
                let spread_y = CELL_SIZE * VERTICAL_DISSIPATION;
                (if dir.x == 0 { CELL_SIZE } else { spread_x })
                    * (if dir.y == 0 { CELL_SIZE } else { spread_x })
                    * (if dir.z == 0 { CELL_HEIGHT } else { spread_y })
            }

            let neighbors: Vec<_> = (0..27)
                .map(|i| Vec3::new(1 - i / 9, 1 - i % 9 / 3, 1 - i % 3))
                .filter(|p| {
                    let p = p + point;

                    (0..self.cells.size().z).contains(&p.z)
                        // Don't interact with cells that are underground
                        && self.consts.get(p.xy()).map(|c| c.alt < CELL_HEIGHT * (p.z + 1) as f64).unwrap_or(true)
                })
                .map(|dir| (dir + point, get_spread_volume(dir)))
                .collect();

            let spread_volume: f64 = neighbors.iter().map(|(_, vol)| vol).sum();
            let cell = self.get_cell(point, time);

            for (p, vol) in neighbors {
                if let Some(c) = swap.get_mut(p) {
                    let part = vol / spread_volume;
                    c.cloud += cell.cloud * part;
                    c.moisture += cell.moisture * part;
                    c.temperature += cell.temperature * part;
                    c.wind += cell.wind * part;
                }
            }
        }
        self.cells
            .iter_mut()
            .for_each(|(_, c)| *c = Cell::default());

        // Wind
        for &point in &points_outer {
            let cell = self.get_cell(point, time);
            let dir_vec = {
                let mut dir = cell.wind;
                dir.apply(|e| e.signum());
                dir.as_::<i32>()
            };

            fn get_spread_volume(dir: Vec3<i32>, vel: Vec3<f64>) -> f64 {
                let vel = vel.map(|e| e.abs());
                (if dir.x == 0 { CELL_SIZE - vel.x } else { vel.x })
                    * (if dir.y == 0 { CELL_SIZE - vel.y } else { vel.y })
                    * (if dir.z == 0 {
                        CELL_HEIGHT - vel.z
                    } else {
                        vel.z
                    })
            }

            let neighbors: Vec<_> = (0..8)
                .map(|i| dir_vec * Vec3::new(i / 4, i / 2 % 2, i % 2))
                .filter(|&p| {
                    let p = p + point;
                    let res = (0..self.cells.size().z).contains(&p.z)
                        // Don't interact with cells that are underground
                        && self.consts.get(p.xy()).map(|c| c.alt < CELL_HEIGHT * (p.z + 1) as f64).unwrap_or(true);
                    res
                })
                .map(|dir| (dir + point, get_spread_volume(dir, cell.wind)))
                .collect();
            let spread_volume: f64 = CELL_SIZE * CELL_SIZE * CELL_HEIGHT;
            let cell = swap
                .get(point)
                .cloned()
                .unwrap_or_else(|| sample_cell(point, time));
            for (p, vol) in neighbors {
                if let Some(c) = self.cells.get_mut(p) {
                    let part = vol / spread_volume;
                    c.cloud += cell.cloud * part;
                    c.moisture += cell.moisture * part;
                    c.temperature += cell.temperature * part;
                    c.wind += cell.wind * part;
                }
            }
        }
        let mut max_temp = 0.0;
        let mut max_wind = Vec3::zero();
        let mut max_moisture = 0.0;
        let mut max_cloud = 0.0;
        let mut max_rain = 0.0;
        let mut max_condens = 0.0;
        for &point in &points {
            // Some variables only apply if the ground is within the cell.
            let grounded = point.z as f64 * CELL_HEIGHT < self.consts[point.xy()].alt;

            self.cells[point].temperature = if grounded {
                f64::lerp(
                    self.cells[point].temperature,
                    self.consts[point.xy()].temp,
                    0.1,
                )
            } else {
                self.cells[point].temperature
            };

            // Wind pull from difference in temperature i.e pressure
            let wind_pull = (0..27)
                .map(|i| Vec3::new(1 - i / 9, 1 - i % 9 / 3, 1 - i % 3))
                .filter(|p| {
                    *p != Vec3::zero() && {
                        let p = p + point;
                        (0..self.cells.size().z).contains(&p.z)
                            && self
                                .consts
                                .get(p.xy())
                                .map(|c| c.alt < CELL_HEIGHT * (p.z + 1) as f64)
                                .unwrap_or(false)
                    }
                })
                .map(|p| {
                    let temp_diff =
                        self.cells[point].temperature - self.cells[p + point].temperature;
                    if (p.z == -1 && temp_diff > 0.0) || (p.z == 1 && temp_diff < 0.0) {
                        Vec3::zero()
                    } else {
                        temp_diff * p.as_().normalized() / 30.0
                    }
                })
                .sum::<Vec3<f64>>();

            // Deflect and apply friction to wind.
            self.cells[point].wind = if grounded {
                let reflect = if self.cells[point].wind.dot(self.consts[point.xy()].normal) < 0.0 {
                    self.cells[point]
                        .wind
                        .reflected(self.consts[point.xy()].normal)
                } else {
                    self.cells[point].wind
                };
                let friction =
                    (self.consts[point.xy()].alt - point.z as f64 * CELL_HEIGHT) / CELL_HEIGHT;

                Vec3::lerp(self.cells[point].wind, reflect, friction * 0.7)
                    * (1.0 - friction * 0.01)
            } else {
                self.cells[point].wind
            } + wind_pull;
            if self.cells[point].wind.magnitude_squared() > CELL_SIZE * CELL_SIZE {
                self.cells[point].wind = self.cells[point].wind.normalized() * CELL_SIZE;
            }

            // Constants NOAA use. https://en.wikipedia.org/wiki/National_Oceanic_and_Atmospheric_Administration
            // There are other sets of constants, might be worth to give them a try
            const B: f64 = 18.678;
            const C: f64 = 257.14;
            // he dew point is the temperature to which air must be cooled to become saturated with water vapor https://en.wikipedia.org/wiki/Dew_point
            let dew_point = (self.cells[point].moisture / 100.0).ln()
                + B * self.cells[point].temperature / (C + self.cells[point].temperature);
            // TODO: convert into a function
            const CRITICAL_UPDRAUGHT: f64 = 20.0;
            let evaporation = if grounded && self.cells[point].temperature >= CRITICAL_UPDRAUGHT {
                self.consts[point.xy()].humidity
                    * (self.cells[point].temperature - CRITICAL_UPDRAUGHT)
            } else {
                0.0
            };

            let condensation = if dew_point >= self.cells[point].temperature {
                // Moisture -> Cloud
                0.025 * self.cells[point].moisture
            } else {
                // Cloud -> Moisture
                -0.05 * self.cells[point].cloud
            };

            const LATENT_MOISTURE_HEAT: f64 = 0.01;
            // Temperature change arising from condensation
            self.cells[point].temperature -= LATENT_MOISTURE_HEAT * condensation;

            self.cells[point].moisture = if grounded {
                f64::lerp(
                    self.cells[point].moisture,
                    self.consts[point.xy()].humidity,
                    0.1,
                )
            } else {
                self.cells[point].moisture
            } + condensation
                + evaporation;

            const RAIN_AMOUNT: f64 = 0.2;
            // At what cloud density it starts raining
            // TODO: Make this a function of temperature?
            const RAIN_CRITICAL: f64 = 0.3;
            let rain = if self.cells[point].cloud > RAIN_CRITICAL {
                RAIN_AMOUNT * (self.cells[point].cloud - RAIN_CRITICAL) / (1.0 - RAIN_CRITICAL)
            } else {
                0.0
            };
            self.cells[point].cloud += condensation - rain;

            self.weather[point].wind = self.cells[point].wind;
            self.weather[point].cloud = self.cells[point].cloud;
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
            if self.weather[point].rain > max_rain {
                max_rain = self.weather[point].rain;
            }
            if condensation < self.consts[point.xy()].temp {
                max_condens = condensation;
            }
        }
        println!("Maxes:\n\tTemperature: {max_temp}\n\tWind: {max_wind}\n\tMoisture: {max_moisture}\n\tCloud: {max_cloud}\n\tRain: {max_rain}\n\tCondensation: {max_condens}");
    }
}
