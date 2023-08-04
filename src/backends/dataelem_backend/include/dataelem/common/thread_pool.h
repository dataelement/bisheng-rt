#ifndef DATAELEM_COMMON_THREADPOOL_H_
#define DATAELEM_COMMON_THREADPOOL_H_

#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dataelem { namespace alg {

class ThreadPool {
 public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();
  size_t size() const { return workers.size(); }

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()> > tasks;
  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) : stop(false)
{
  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(
              lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty())
            return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }
        task();
      }
    });
}

// add new work item to the pool
template <class F, class... Args>
auto
ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
  using return_type = typename std::result_of<F(Args...)>::type;
  auto task = std::make_shared<std::packaged_task<return_type()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");
    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) worker.join();
}

template <typename T>
inline void
GetAsyncRets(std::vector<T>& rets)
{
  for (auto& ret : rets) {
    ret.get();
  }
}

inline ThreadPool&
nn_thread_pool()
{
  static ThreadPool tp(std::thread::hardware_concurrency());
  return tp;
}

inline size_t
nn_concurrency()
{
  return nn_thread_pool().size();
}

typedef std::future<bool> BoolFuture;

template <typename FUNC>
void
parallel_run(FUNC func, ThreadPool& tp = nn_thread_pool())
{
  std::vector<BoolFuture> rets(tp.size());
  for (size_t i = 0; i < rets.size(); ++i) {
    rets[i] = tp.enqueue([&func, i]() {
      func(i);
      return true;
    });
  }
  for (auto& ret : rets) {
    ret.get();
  }
}

template <typename FUNC>
void
parallel_run_range(size_t n, FUNC func, ThreadPool& tp = nn_thread_pool())
{
  std::vector<BoolFuture> rets(tp.size());
  size_t concurrency = rets.size();
  for (size_t i = 0; i < rets.size(); ++i) {
    size_t start = i * n / concurrency;
    size_t end = std::min((i + 1) * n / concurrency, n);
    rets[i] = tp.enqueue([&func, i, start, end]() {
      func(i, start, end);
      return true;
    });
  }
  for (auto& ret : rets) {
    ret.get();
  }
}

template <typename FUNC>
void
parallel_run_dynamic(size_t n, FUNC func, ThreadPool& tp = nn_thread_pool())
{
  std::vector<BoolFuture> rets(n);
  for (size_t i = 0; i < n; ++i) {
    rets[i] = tp.enqueue([&func, i]() {
      func(i);
      return true;
    });
  }
  for (auto& ret : rets) {
    ret.get();
  }
}

}}  // namespace dataelem::alg

#endif  // DATAELEM_COMMON_THREADPOOL_H_