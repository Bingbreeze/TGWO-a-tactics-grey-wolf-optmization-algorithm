# TGWO

TGWO（Tactical Grey Wolf Optimizer）是一个基于灰狼优化框架的群智能优化算法实现。

本目录只包含 TGWO 算法源码，不包含 CEC 基准函数、对比算法、实验结果或 MATLAB 二进制文件。

## 文件

```text
TGWO.m
README.md
LICENSE
```

## 调用格式

```matlab
[bestScore, bestPosition, convergenceCurve] = ...
    TGWO(nPop, MaxFEs, lb, ub, dim, fobj)
```

### 输入参数

- `nPop`：种群规模。
- `MaxFEs`：最大函数评价次数。
- `lb`、`ub`：搜索边界，可以是标量，也可以是长度为 `dim` 的行向量。
- `dim`：问题维度。
- `fobj`：目标函数句柄，输入一个 `1 x dim` 行向量并返回标量适应度。

还支持三个可选参数：

```matlab
TGWO(nPop, MaxFEs, lb, ub, dim, fobj, para_k, para_c, para_p)
```

- `para_k`：概率调度参数，默认值为 `15`。
- `para_c`：概率调度中心参数，默认值为 `0.5`。
- `para_p`：精英个体比例，默认值为 `0.2`。

### 输出参数

- `bestScore`：搜索得到的最优适应度。
- `bestPosition`：对应的最优位置。
- `convergenceCurve`：长度为 `MaxFEs` 的收敛曲线。

## 最小示例

将 MATLAB 当前目录切换到本目录后运行：

```matlab
clc; clear;
nPop = 30;
MaxFEs = 3000;
dim = 10;
lb = -100;
ub = 100;
fobj = @(x) sum(x.^2);

[bestScore, bestPosition, convergenceCurve] = ...
    TGWO(nPop, MaxFEs, lb, ub, dim, fobj);

fprintf('Best score: %.6g\n', bestScore);
plot(convergenceCurve, 'LineWidth', 1.5);
xlabel('FEs');
ylabel('Best fitness');
grid on;
```

## 许可证

本项目使用 MIT License，具体条款见 `LICENSE`。
