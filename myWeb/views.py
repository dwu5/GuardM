import os

from django.contrib import messages
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect

from complexity_exp.wm_main import wm_main
from . import models, forms


# 欢迎
def welcome(request):
    return render(request, 'welcome.html')


# 首页
def index(request):
    if not request.session.get('is_login', None):
        return redirect('/login/')
    return render(request, 'index.html')


# 用户登录
def login(request):
    if request.session.get('is_login', None):  # 不允许重复登录
        return redirect('/index/')
    if request.method == "POST":
        login_form = forms.UserForm(request.POST)
        message = '请检查填写的内容！'
        if login_form.is_valid():
            username = login_form.cleaned_data.get('username')
            password = login_form.cleaned_data.get('password')

            try:
                user = models.User.objects.get(name=username)
            except:
                message = '用户不存在！'
                return render(request, 'login.html', locals())

            if user.password == password:
                request.session['is_login'] = True  # 将用户状态和数据写入session字典
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                print("登录成功")
                print("username:" + username)
                return redirect('/index/')
            else:
                message = '密码不正确！'
                return render(request, 'login.html', locals())
        else:
            return render(request, 'login.html', locals())

    login_form = forms.UserForm()
    return render(request, 'login.html', locals())


# 用户注册
def register(request):
    if request.session.get('is_login', None):
        return redirect('/index/')

    if request.method == 'POST':
        register_form = forms.RegisterForm(request.POST)
        message = "请检查填写的内容！"
        if register_form.is_valid():
            username = register_form.cleaned_data.get('username')
            password1 = register_form.cleaned_data.get('password1')
            password2 = register_form.cleaned_data.get('password2')
            email = register_form.cleaned_data.get('email')

            if password1 != password2:
                message = '两次输入的密码不同！'
                return render(request, 'register.html', locals())
            else:
                same_name_user = models.User.objects.filter(name=username)
                if same_name_user:
                    message = '用户名已经存在'
                    return render(request, 'register.html', locals())
                same_email_user = models.User.objects.filter(email=email)
                if same_email_user:
                    message = '该邮箱已经被注册了！'
                    return render(request, 'register.html', locals())

                new_user = models.User()
                new_user.name = username
                new_user.password = password1
                new_user.email = email
                new_user.save()

                messages.success(request, "注册成功！")

                return redirect('/login/')
        else:
            return render(request, 'register.html', locals())

    register_form = forms.RegisterForm()
    return render(request, 'register.html', locals())


# 用户退出
def logout(request):
    if not request.session.get('is_login', None):
        return redirect("/welcome/")
    request.session.flush()  # 一次性将session中的所有内容全部清空
    return redirect("/welcome/")


# 跳转至文件提交界面
def index_use(request):
    if not request.session.get('is_login', None):
        return redirect("/login/")
    return render(request, 'upload.html', locals())


# 用户上传文件
def upload(request):
    if not request.session.get('is_login', None):
        return redirect("/login/")

    if request.method == 'POST':
        upload_file = request.FILES.get("file")

        username = request.session['user_name']
        user = models.User.objects.get(name=username)
        user.file = upload_file  # 将上传文件提交至后台用户表
        user.status = 'B'  # 更新用户服务进度
        user.save()

        # path = 'myWeb/userFile/'
        path = './media/'
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, upload_file.name)
        with open(path, 'wb') as f:
            for i in upload_file:
                f.write(i)

        print(path)
        print("用户" + username + "上传文件")
        print(request.FILES.get('file'))

        messages.success(request, "上传成功！")

    return render(request, 'index.html', locals())


# 后台启动训练
def start_training(request):
    messages.success(request, "开始训练！")
    wm_main()
    return render(request, 'temp.html', locals())


# 查看训练进度
def index_to_progress(request):
    username = request.session['user_name']
    user = models.User.objects.get(name=username)
    status = user.get_status_display()  # 显示用户服务进度

    file_name = user.file
    print('This user has uploaded :', file_name)

    from complexity_exp.wm_main import myepoch1
    if username in myepoch1.keys() and status != 'Model uploaded, training completed':
        user.epoch = myepoch1[username]
        show_epoch = user.epoch
        user.save()
        print("username: ", user.name)
        print("user.epoch: ", user.epoch)

    show_epoch = user.epoch
    print("username: ", user.name)
    print("user.epoch", user.epoch)

    return render(request, 'progress.html', locals())


# 训练结果可视化
def visualization(request):
    return render(request, 'visualization.html', locals())
