# CUDA Ray Tracing



## Описание программы

Программа принимает на вход количество сфер, количество источников света,
размер изображения и имя файла для сохранения. 

```
PS C:\Users\Admin\source\repos\George909\ray_tracing_cuda\x64\Release> .\rae_tracing_cuda.exe 10 5 1900x1900 img
```

После этого создается 3D сцена с указанным количеством сфер и источников 
и производится трассировка лучей на GPU и CPU.

Программа выводит время потраченное на трасировку лучей.

```
PS C:\Users\Admin\source\repos\George909\ray_tracing_cuda\x64\Release> .\rae_tracing_cuda.exe 10 5 1900x1900 img
Time CPU: 1088 ms
Time GPU: 116 ms
```

## Пример работы

```
PS C:\Users\Admin\source\repos\George909\ray_tracing_cuda\x64\Release> .\rae_tracing_cuda.exe 5 3 1200x600 img
Time CPU: 76 ms
Time GPU: 78 ms
```

![Иллюстрация к проекту](https://github.com/George909/ray_tracing_cuda/blob/master/rae_tracing_cuda/img/img_1200x600.BMP)

```
PS C:\Users\Admin\source\repos\George909\ray_tracing_cuda\x64\Release> .\rae_tracing_cuda.exe 10 5 1900x1900 img
Time CPU: 823 ms
Time GPU: 100 ms
```

![Иллюстрация к проекту](https://github.com/George909/ray_tracing_cuda/blob/master/rae_tracing_cuda/img/img_1900x1900.BMP)



## Конфигурация системы

|     |                               |
|-----|-------------------------------|
| CPU | Intel Core i7-3770 CPU 3.40GHz|
| GPU | NVIDIA GeForce GTX 770        |
| RAM | 16 Gb                         |
| OS  | Windows 10 Proffesional       | 


## GPU vs CPU

|Количество сфер|Количество источников света|Размер изображения|CPU, ms|GPU, ms|CPU/GPU|
|---------------|---------------------------|------------------|-------|-------|-------|
|       10      |         5                 |      600x600     | 106   |   81  | 1.30  |
|       10      |         5                 |      900x900     | 146   |   78  | 1.87  |
|       10      |         5                 |      1200x1200   | 458   |   84  | 5.45  |
|       10      |         5                 |      1500x1500   | 731   |   90  | 8.12  |
|       10      |         5                 |      1400x1400   | 909   |   90  | 10.1  |


