use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

#[allow(dead_code)]
pub struct MemoryPool<T> {
    pool: Arc<Mutex<VecDeque<T>>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
}

#[allow(dead_code)]
impl<T> MemoryPool<T> {
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::new())),
            factory: Box::new(factory),
        }
    }

    pub async fn acquire(&self) -> T {
        let mut pool = self.pool.lock().await;
        if let Some(item) = pool.pop_front() {
            item
        } else {
            (self.factory)()
        }
    }

    pub async fn release(&self, item: T) {
        let mut pool = self.pool.lock().await;
        pool.push_back(item);
    }

    pub async fn size(&self) -> usize {
        let pool = self.pool.lock().await;
        pool.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_pool_acquire_and_release() {
        let pool = MemoryPool::new(|| Vec::<u8>::with_capacity(1024));
        assert_eq!(pool.size().await, 0);

        let mut item = pool.acquire().await;
        assert_eq!(item.capacity(), 1024);
        item.push(1);
        item.clear();

        pool.release(item).await;
        assert_eq!(pool.size().await, 1);

        let reused = pool.acquire().await;
        assert_eq!(reused.capacity(), 1024);
        assert_eq!(pool.size().await, 0);
    }
}
