# GuardM
![](myWeb/static/welcome/image/demo.png)

A simple demo of a deep learning model protection system.

Actually, it's the back-end module of the entire protection system. Its focus is to demonstrate the procedure of model protection business we set, not the algorithm itself. 

If you are interested in how we implement the deep learning algorithm, please refer to [our work](https://github.com/ByGary/Security-of-IP-Protection-Frameworks).

## Get started
`python manage.py makemigrations` to create a model and record data migration.

`python manage.py migrate` to update dataset.

`python manage.py createsuperuser` to create super user (administrator).

`python manage.py runserver` to start the built-in web server.

## Progress
2022年2月2日10:37:00

登录注册子系统（首页创建未设计，登录无验证码，注册无邮箱验证）；

2022年2月6日01:00:01

登录注册子系统；用户上传文件功能，后台更新用户表；

2022年2月11日00:10:54

admin自定义按钮；

2022年2月12日16:50:38

水印代码complexity_exp(views.py调用水印代码main函数，报错)；

2022年2月15日15:22:58

后台启动complexity_exp训练；

2022年2月21日16:28:57

用户服务状态显示与更新；

2022年2月22日20:32:24

导航栏，欢迎页面，按钮；

2022年2月23日23:55:05

首页；

2022年2月28日15:29:34

用户增加epoch属性（未绑定complexity_exp中的epoch与用户属性的epoch）；后台增加action启动训练；

2022年3月2日19:03:19

绑定complexity_exp中的epoch与用户属性的epoch

2022年3月10日14:49:25

结果可视化

2022年3月14日20:18:59

前台增加用户上传文件名显示

2022年3月15日22:34:48

后台用户部分属性设置为不可修改

2022年4月18日21:57:11

TensorBoard可视化
