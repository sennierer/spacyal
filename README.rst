Spacy Active learning
====================

.. image:: https://zenodo.org/badge/130271493.svg
   :target: https://zenodo.org/badge/latestdoi/130271493

Django app that uses active learning (deliberately picking the examples to annotate) to retrain the spaCy_ NER module more effectively.

Prerequisites
-------------

For spacyal to run you need a working Celery_ installation. Something along the lines of::

  from __future__ import absolute_import, unicode_literals
  import os
  from celery import Celery

  app = Celery('tasks')

  # Using a string here means the worker doesn't have to serialize
  # the configuration object to child processes.
  # - namespace='CELERY' means all celery-related configuration keys
  #   should have a `CELERY_` prefix.
  app.config_from_object('django.conf:settings', namespace='CELERY')

  # Load task modules from all registered Django app configs.
  app.autodiscover_tasks()


  @app.task(bind=True)
  def debug_task(self):
      print('Request: {0!r}'.format(self.request))


Installation
------------

* Install the package
* include spacyal.urls and spacyal.api_urls in your main url definition
* ensure that you have a base template called base.html
* run python manage.py migrate
* and you should be good to go


.. _Celery: http://www.celeryproject.org/
.. _spaCy: https://www.spacy.io
