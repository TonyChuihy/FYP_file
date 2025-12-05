import platform
import os

def get_cpu_info():
    """获取CPU基本信息"""
    print("=== CPU 信息 ===")
    
    # 系统信息
    print(f"系统: {platform.system()}")
    print(f"平台: {platform.platform()}")
    print(f"处理器: {platform.processor()}")
    print(f"架构: {platform.machine()}")
    
    # 更详细的CPU信息
    try:
        import subprocess
        if platform.system() == "Windows":
            result = subprocess.run(['wmic', 'cpu', 'get', 'name'], capture_output=True, text=True)
            cpu_name = result.stdout.strip().split('\n')[1] if result.returncode == 0 else "Unknown"
        elif platform.system() == "Linux":
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_name = line.split(':')[1].strip()
                        break
                else:
                    cpu_name = "Unknown"
        else:  # macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True)
            cpu_name = result.stdout.strip() if result.returncode == 0 else "Unknown"
        
        print(f"CPU型号: {cpu_name}")
        
    except Exception as e:
        print(f"获取CPU详细信息失败: {e}")

get_cpu_info()