from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
]

urlpatterns += [
    path('video_details.html', views.video_details),
]

urlpatterns += [
    path('show_video_info.html', views.show_video_info),
]