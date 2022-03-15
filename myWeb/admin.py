from django.contrib import admin, messages

# Register your models here.
from django.shortcuts import render
from django.utils.html import format_html

from complexity_exp.wm_main import wm_main
from . import models, forms
from .models import User


# # 创建ModelAdmin的子类
# class UserAdmin(admin.ModelAdmin):
#     pass
#
#
# # 注册的时候，将原模型和ModelAdmin耦合起来
# admin.site.register(User, UserAdmin)

admin.site.site_header = 'GuardM'
admin.site.index_title = '后台管理'


def starter(self):
    return format_html(
        '<a href="/start_training/">启动<a/>'
    )


def training_start_switch(modeladmin, request, queryset):
    username = queryset[0]
    user = models.User.objects.get(name=username)
    user.status = 'C'
    user.save()

    wm_main()
    messages.success(request, "开始训练！")
    return render(request, 'draft.html', locals())


def set_model_not_uploaded(modeladmin, request, queryset):
    print(queryset)
    queryset.update(status='A')


def set_training_not_started(modeladmin, request, queryset):
    print(queryset)
    queryset.update(status='B')


def set_training_in_progress(modeladmin, request, queryset):
    print(queryset)
    queryset.update(status='C')


def set_training_completed(modeladmin, request, queryset):
    print(queryset)
    queryset.update(status='D')


starter.short_description = '训练'
training_start_switch.short_description = "Training Start"
set_model_not_uploaded.short_description = "Mark selected users as Model not uploaded"
set_training_not_started.short_description = "Mark selected users as Training not started"
set_training_in_progress.short_description = "Mark selected users as Training in progress"
set_training_completed.short_description = "Mark selected users as Training completed"


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    readonly_fields = ("name", "password", "email", "file",)
    list_display = ('name', 'email', 'c_time', 'status', 'file')
    actions = [training_start_switch, set_model_not_uploaded, set_training_not_started, set_training_in_progress,
               set_training_completed]
