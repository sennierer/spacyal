from rest_framework.views import APIView
from .models import al_project, case
from spacyal.tasks import get_cases, retrain_model
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from django_celery_results.models import TaskResult
import json, os
from rest_framework import permissions
from rest_framework.exceptions import PermissionDenied
from shutil import make_archive
from wsgiref.util import FileWrapper
import datetime


class UserALProjectPerm(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        print(obj)
        if request.user.is_superuser:
            return True
        elif request.user in obj.users_allowed.all():
            return True
        raise PermissionDenied({"message": "You don't have permission to"
                                " work on this Project",
                                "object_id": obj.id})


class RetrieveCasesView(APIView):
    """view that allows to retrieve new cases related to a project"""
    permission_classes = (UserALProjectPerm,)

    def get(self, request):
        project_id = request.query_params.get('project_id', None)
        self.check_object_permissions(
            request, al_project.objects.get(pk=project_id))
        t = TaskResult.objects.filter(
            status='SUCCESS',
            result__icontains='"project": {}}}'.format(project_id)).order_by('-id')
        if t.count() > 0:
            c = json.loads(t[0].result)
            get_cases.delay(
                project_id, model=c['folder'], retrained=c['retrained'])
        else:
            get_cases.delay(project_id, retrained=False)
        c = case.objects.filter(project_id=project_id, decission__isnull=True)
        res = [obj.as_dict() for obj in c]
        return Response(res)

    def post(self, request):
        case_id = request.data['case_id']
        decission = request.data['decission']
        retrain = request.data.get('retrain', False)
        project_id = request.data.get('project_id', None)
        c = case.objects.get(pk=case_id)
        c.decission = decission
        c.user = request.user
        c.use = True
        c.save()
        res = {'status': 'saved', 'm_hash': False}
        if retrain:
            t = TaskResult.objects.filter(
                status='SUCCESS',
                result__icontains='"project": {}}}'.format(project_id)).order_by('-id')
            if t.count() > 0:
                c = json.loads(t[0].result)
                m = c['folder'].split('/')[-1]
            else:
                m = 'model_1'
            res['m_hash'] = retrain_model.delay(project_id, m).id
        return Response(res, status=status.HTTP_201_CREATED)


class GetProgressModelView(APIView):

    def get(self, request):
        celery_task = request.query_params.get('celery_task')
        t = TaskResult.objects.get(task_id=celery_task)
        if t.status == 'PROGRESS':
            r = {'status': 'PROGRESS', 'percent': json.loads(t.result)['progress']}
        elif t.status == 'SUCCESS':
            r = {'status': 'SUCCESS', 'percent': 100}
        elif t.status == 'FAILURE':
            r = {'status': 'FAILURE', 'message': t.result}
        return Response(r)


class DownloadModelView(APIView):

    def get(self, request):
        project_id = request.query_params.get('project_id', None)
        project = al_project.objects.get(pk=project_id)
        self.check_object_permissions(
            request, project)
        base_d = '/'.join(project.texts.path.split('/')[:-1])
        t = TaskResult.objects.filter(
            status='SUCCESS',
            result__icontains='"project": {}}}'.format(project_id)).order_by('-id')
        if t.count() > 0:
            c = json.loads(t[0].result)
            model_name = c['folder'].split('/')[-1]
            ts = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            filename = "{}_{}_{}".format(project.name, model_name, ts)
            export_file = make_archive(os.path.join(base_d, filename), 'zip',
                                       base_d, model_name)
            response = HttpResponse(FileWrapper(open(export_file, 'rb')),
                                    content_type='application/zip')
            response['Content-Disposition'] = 'attachment; filename="{}.zip"'.format(
                filename)
            return response


class DownloadCasesView(APIView):

    def get(self, request):
        choices = ('skip', 'correct', 'wrong')
        project_id = request.query_params.get('project_id', None)
        exclude_wrong = request.query_params.get('exclude_wrong', False)
        if exclude_wrong:
            res = al_project.objects.get(pk=project_id).get_training_data(
                include_all=True, include_negative=False)
            for idx, e in enumerate(res):
                res[idx][1]['entities'] = [(x[0], x[1], x[2]) for x in e[1]['entities']]
        else:
            res = al_project.objects.get(pk=project_id).get_training_data(
                include_all=True, include_negative=True)
            for idx, e in enumerate(res):
                res[idx][1]['entities'] = [(x[0], x[1], x[2], choices[x[3]])
                                           for x in e[1]['entities']]
        return Response(res)
