from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("search/", views.search, name="search"),
    path("doc/<int:pk>/", views.view_doc, name="view-doc"),
]
