import subprocess
import sys

def test_main_mpc():
    """测试主程序的MPC选项"""
    print("=== 测试主程序MPC选项 ===")
    
    try:
        # 模拟用户输入：选择A*算法(2) + MPC控制器(2)
        input_data = "2\n2\n"
        
        # 运行主程序，设置超时避免无限等待
        result = subprocess.run(
            [sys.executable, "main.py"],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=60  # 60秒超时
        )
        
        print("程序输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ 主程序MPC测试成功！")
            return True
        else:
            print(f"❌ 主程序返回错误代码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 程序运行超时（60秒）")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    success = test_main_mpc()
    if success:
        print("\n🎯 主程序MPC集成测试成功！")
    else:
        print("\n💥 主程序MPC集成测试失败...") 