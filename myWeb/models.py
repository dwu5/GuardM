from django.db import models
from django import forms


# Create your models here.
class User(models.Model):
    STATUS_CHOICES = (
        ('A', 'Model not uploaded'),
        ('B', 'Model uploaded, training not started'),
        ('C', 'Model uploaded, training in progress'),
        ('D', 'Model uploaded, training completed')
    )

    name = models.CharField('用户名', max_length=128, unique=True)
    password = models.CharField('密码', max_length=256)
    email = models.EmailField('邮箱', unique=True)
    c_time = models.DateTimeField('创建时间', auto_now_add=True)
    file = models.FileField('文件', upload_to='', null=False, blank=False)
    status = models.CharField('状态', max_length=256, choices=STATUS_CHOICES, default='A')
    epoch = models.IntegerField('epoch', default=0)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["-c_time"]
        verbose_name = "用户"
        verbose_name_plural = "用户"
