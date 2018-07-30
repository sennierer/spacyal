import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


setup(
    name='acdh-spacyal',
    version='0.3.2',
    packages=find_packages(
        exclude=['spacyal/__pycache__']),
    include_package_data=True,
    license='MIT License',  # example license
    description='Addon to Django apps that allows to retrain Spacy NER with active learning.',
    long_description=README,
    url='https://github.com/sennierer/spacyal',
    author='Matthias Schlögl',
    author_email='matthias.schloegl@oeaw.ac.at',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Web Environment',
        'Framework :: Django',
        'Framework :: Django :: 2.0',  # replace "X.Y" as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # example license
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
    ],
    install_requires=[
        'celery>=4.1.0',
        'amqp==2.2.2',
        'Django>=2.0.4,<3',
        'django-celery-results>=1.0.1',
        'django-crispy-forms>=1.7.2',
        'django-extensions>=2.0.6',
        'djangorestframework>=3.8.0',
        'pandas>=0.22.0',
        'spacy>=2.0.10',
        'django-tables2>=1.21.2'
    ]
)
