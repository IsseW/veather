use std::ops::{Index, IndexMut};
use vek::*;

#[derive(Debug, Clone)]
pub struct Grid<T> {
    cells: Vec<T>,
    size: Vec2<i32>, // TODO: use u32
}

impl<T> Grid<T> {
    pub fn from_raw(size: Vec2<i32>, raw: impl Into<Vec<T>>) -> Self {
        let cells = raw.into();
        assert_eq!(size.product() as usize, cells.len());
        Self { cells, size }
    }

    pub fn populate_from(size: Vec2<i32>, mut f: impl FnMut(Vec2<i32>) -> T) -> Self {
        Self {
            cells: (0..size.y)
                .map(|y| (0..size.x).map(move |x| Vec2::new(x, y)))
                .flatten()
                .map(&mut f)
                .collect(),
            size,
        }
    }

    pub fn new(size: Vec2<i32>, default_cell: T) -> Self
    where
        T: Clone,
    {
        Self {
            cells: vec![default_cell; size.product() as usize],
            size,
        }
    }

    fn idx(&self, pos: Vec2<i32>) -> Option<usize> {
        if pos.map2(self.size, |e, sz| e >= 0 && e < sz).reduce_and() {
            Some((pos.y * self.size.x + pos.x) as usize)
        } else {
            None
        }
    }

    pub fn size(&self) -> Vec2<i32> {
        self.size
    }

    pub fn get(&self, pos: Vec2<i32>) -> Option<&T> {
        self.cells.get(self.idx(pos)?)
    }

    pub fn get_mut(&mut self, pos: Vec2<i32>) -> Option<&mut T> {
        let idx = self.idx(pos)?;
        self.cells.get_mut(idx)
    }

    pub fn set(&mut self, pos: Vec2<i32>, cell: T) -> Option<T> {
        let idx = self.idx(pos)?;
        self.cells.get_mut(idx).map(|c| core::mem::replace(c, cell))
    }

    pub fn iter(&self) -> impl Iterator<Item = (Vec2<i32>, &T)> + '_ {
        let w = self.size.x;
        self.cells
            .iter()
            .enumerate()
            .map(move |(i, cell)| (Vec2::new(i as i32 % w, i as i32 / w), cell))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Vec2<i32>, &mut T)> + '_ {
        let w = self.size.x;
        self.cells
            .iter_mut()
            .enumerate()
            .map(move |(i, cell)| (Vec2::new(i as i32 % w, i as i32 / w), cell))
    }

    pub fn iter_area(
        &self,
        pos: Vec2<i32>,
        size: Vec2<i32>,
    ) -> impl Iterator<Item = Option<(Vec2<i32>, &T)>> + '_ {
        (0..size.x)
            .map(move |x| {
                (0..size.y).map(move |y| {
                    Some((
                        pos + Vec2::new(x, y),
                        &self.cells[self.idx(pos + Vec2::new(x, y))?],
                    ))
                })
            })
            .flatten()
    }

    pub fn raw(&self) -> &[T] {
        &self.cells
    }
}

impl<T> Index<Vec2<i32>> for Grid<T> {
    type Output = T;

    fn index(&self, index: Vec2<i32>) -> &Self::Output {
        self.get(index).unwrap_or_else(|| {
            panic!(
                "Attempted to index grid of size {:?} with index {:?}",
                self.size(),
                index
            )
        })
    }
}

impl<T> IndexMut<Vec2<i32>> for Grid<T> {
    fn index_mut(&mut self, index: Vec2<i32>) -> &mut Self::Output {
        let size = self.size();
        self.get_mut(index).unwrap_or_else(|| {
            panic!(
                "Attempted to index grid of size {:?} with index {:?}",
                size, index
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct Volume<T> {
    cells: Vec<T>,
    size: Vec3<i32>, // TODO: use u32
}

impl<T> Volume<T> {
    pub fn from_raw(size: Vec3<i32>, raw: impl Into<Vec<T>>) -> Self {
        let cells = raw.into();
        assert_eq!(size.product() as usize, cells.len());
        Self { cells, size }
    }

    pub fn populate_from(size: Vec3<i32>, mut f: impl FnMut(Vec3<i32>) -> T) -> Self {
        Self {
            cells: (0..size.z)
                .map(|z| {
                    (0..size.y)
                        .map(move |y| (0..size.x).map(move |x| Vec3::new(x, y, z)))
                        .flatten()
                })
                .flatten()
                .map(&mut f)
                .collect(),
            size,
        }
    }

    pub fn new(size: Vec3<i32>, default_cell: T) -> Self
    where
        T: Clone,
    {
        Self {
            cells: vec![default_cell; size.product() as usize],
            size,
        }
    }

    fn idx(&self, pos: Vec3<i32>) -> Option<usize> {
        if pos.map2(self.size, |e, sz| e >= 0 && e < sz).reduce_and() {
            Some((pos.z * self.size.x * self.size.y + pos.y * self.size.x + pos.x) as usize)
        } else {
            None
        }
    }

    pub fn size(&self) -> Vec3<i32> {
        self.size
    }

    pub fn get(&self, pos: Vec3<i32>) -> Option<&T> {
        self.cells.get(self.idx(pos)?)
    }

    pub fn get_mut(&mut self, pos: Vec3<i32>) -> Option<&mut T> {
        let idx = self.idx(pos)?;
        self.cells.get_mut(idx)
    }

    pub fn set(&mut self, pos: Vec3<i32>, cell: T) -> Option<T> {
        let idx = self.idx(pos)?;
        self.cells.get_mut(idx).map(|c| core::mem::replace(c, cell))
    }

    pub fn iter(&self) -> impl Iterator<Item = (Vec3<i32>, &T)> + '_ {
        let w = self.size.x;
        let h = self.size.y;
        self.cells.iter().enumerate().map(move |(i, cell)| {
            (
                Vec3::new(i as i32 % w, i as i32 / w % h, i as i32 / w / h),
                cell,
            )
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Vec3<i32>, &mut T)> + '_ {
        let w = self.size.x;
        let h = self.size.y;
        self.cells.iter_mut().enumerate().map(move |(i, cell)| {
            (
                Vec3::new(i as i32 % w, i as i32 / w % h, i as i32 / h / w),
                cell,
            )
        })
    }

    pub fn iter_index(&self) -> impl Iterator<Item = Vec3<i32>> + '_ {
        (0..self.size.z)
            .map(move |z| {
                (0..self.size.y)
                    .map(move |y| (0..self.size.x).map(move |x| Vec3::new(x, y, z)))
                    .flatten()
            })
            .flatten()
    }

    pub fn iter_volume(
        &self,
        pos: Vec3<i32>,
        size: Vec3<i32>,
    ) -> impl Iterator<Item = Option<(Vec3<i32>, &T)>> + '_ {
        (0..size.z)
            .map(move |z| {
                (0..size.y)
                    .map(move |y| {
                        (0..size.x).map(move |x| {
                            Some((
                                pos + Vec3::new(x, y, z),
                                &self.cells[self.idx(pos + Vec3::new(x, y, z))?],
                            ))
                        })
                    })
                    .flatten()
            })
            .flatten()
    }

    pub fn raw(&self) -> &[T] {
        &self.cells
    }
}

impl<T> Index<Vec3<i32>> for Volume<T> {
    type Output = T;

    fn index(&self, index: Vec3<i32>) -> &Self::Output {
        self.get(index).unwrap_or_else(|| {
            panic!(
                "Attempted to index grid of size {:?} with index {:?}",
                self.size(),
                index
            )
        })
    }
}

impl<T> IndexMut<Vec3<i32>> for Volume<T> {
    fn index_mut(&mut self, index: Vec3<i32>) -> &mut Self::Output {
        let size = self.size();
        self.get_mut(index).unwrap_or_else(|| {
            panic!(
                "Attempted to index grid of size {:?} with index {:?}",
                size, index
            )
        })
    }
}
