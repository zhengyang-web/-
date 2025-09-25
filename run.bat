@echo off

echo 正在安装依赖...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo 依赖安装失败，请检查网络连接或Python环境！
    pause
    exit /b %errorlevel%
)

echo 依赖安装成功！

:menu
cls
echo ===== 全连接神经网络程序菜单 =====
echo 1. 运行主程序 (MNIST手写数字识别)
echo 2. 运行扩展功能演示
echo 3. 运行正确分类演示
echo 4. 演示所有程序

echo 请选择要运行的程序: 
set /p choice=

if %choice% == 1 (
    echo 正在运行主程序...
    python main.py
) elif %choice% == 2 (
    echo 正在运行扩展功能演示...
    python demo_extensions.py
) elif %choice% == 3 (
    echo 正在运行正确分类演示...
    python demo_correct_classification.py
) elif %choice% == 4 (
    echo 开始演示所有程序...
    echo. & echo ===== 1. 运行主程序 (MNIST手写数字识别) =====
    python main.py
    echo. & echo ===== 2. 运行扩展功能演示 =====
    python demo_extensions.py
    echo. & echo ===== 3. 运行正确分类演示 =====
    python demo_correct_classification.py
    echo. & echo ===== 所有程序演示完成 =====
) else (
    echo 无效的选择，请重新输入
    pause
    goto menu
)

if %errorlevel% neq 0 (
    echo 程序运行失败！
    pause
    exit /b %errorlevel%
)

pause