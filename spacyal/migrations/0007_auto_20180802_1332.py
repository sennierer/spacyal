# Generated by Django 2.0.4 on 2018-08-02 13:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('spacyal', '0006_auto_20180426_1356'),
    ]

    operations = [
        migrations.AddField(
            model_name='al_project',
            name='threshold_probability',
            field=models.FloatField(blank=True, null=True, verbose_name='probability threshold'),
        ),
        migrations.AlterField(
            model_name='al_project',
            name='num_retrain',
            field=models.PositiveSmallIntegerField(default=40, verbose_name='number of examples before retraining'),
        ),
    ]
