import os

def max_threads():
    # 获取可用的最大线程数
    try:
        return os.cpu_count() or 1  # 返回CPU核心数，若无法获取则返回1
    except Exception as e:
        print(f"获取最大线程数时出错: {e}")
        return 1

if __name__ == "__main__":
    max_thread_count = max_threads()
    print(f"最大可用线程数: {max_thread_count}")
