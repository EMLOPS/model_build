
"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn import SKLearn, SKLearnProcessor
from sagemaker.processing import FrameworkProcessor
from sagemaker.pytorch.processing import PyTorchProcessor

from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))




def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

list_files(os.getcwd())
print('@@@@####$$$$ CWD---', os.getcwd())

def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="IntelPackageGroup",
    pipeline_name="IntelPipeline",
    base_job_prefix="Intel",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on intel data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # [START] Intel pipeline
    
    dvc_repo_url = ParameterString(
        name="DVCRepoURL",
        default_value="codecommit::ap-south-1://sagemaker_intel_image"
    )
    dvc_branch = ParameterString(
        name="DVCBranch",
        default_value="pipeline-processed-dataset"
    )
    
    input_dataset = ParameterString(
        name="InputDatasetZip",
        default_value="s3://sagemaker-ap-south-1-358292687379/intel-image.zip"
    )
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    
    # model = ParameterString(
    #     name="MODEL", default_value="resnet18"
    # )
    model_name = ParameterString(
        name="MODEL", default_value="efficientnet_b0"
    )

    optimizer = ParameterString(
        name="Optimizer", default_value="ADAM"
    )

    batchsize = ParameterString(
        name="BatchSize", default_value="128"
    ) 

    learningrate = ParameterString(
        name="LearningRate", default_value="0.00007"
    ) 

    epochs = ParameterString(
        name="EPOCHS", default_value="1"
    ) 
    
    base_job_name = base_job_prefix
    
    # PREPROCESS STEP

    sklearn_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="0.23-1",
        # instance_type="ml.t3.medium",
        instance_type="ml.m5.xlarge",
        # instance_type='local',

        # Spot training
        # use_spot_instances=True,
        # max_wait=1800,
        # max_run=2000,

        instance_count=1,
        base_job_name=f"{base_job_name}-preprocess-intel-dataset",
        sagemaker_session=pipeline_session,
        # sagemaker_session=local_pipeline_session,
        role=role,
        env={
            "DVC_REPO_URL": dvc_repo_url,
            "DVC_BRANCH": dvc_branch,
            # "DVC_REPO_URL": "codecommit::ap-south-1://sagemaker_intel_image",
            # "DVC_BRANCH": "processed-dataset-pipeline",
            "GIT_USER": "Nikhil Shrimali",
            "GIT_EMAIL": "nshrimali21@gmail.com"
        }
    )
    print('@@@@@@@@@@@ Check if this path exists - ', os.path.exists(os.path.join(os.getcwd(), "pipelines/intel/sagemaker-intel/preprocess.py")))
    
    processing_step_args = sklearn_processor.run(
        code='preprocess.py',
        source_dir= os.path.join(os.getcwd(), "pipelines/intel/sagemaker-intel"),
        # dependencies="sagemaker-flower-pipeline/requirements.txt",
        inputs=[
            ProcessingInput(
                input_name='data',
                source=input_dataset,
                # source="s3://sagemaker-ap-south-1-358292687379/intel-image.zip",
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/dataset/train"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/dataset/test"
            ),
        ],
    )
    
    step_process = ProcessingStep(
        name="PreprocessIntelClassifierDataset",
        step_args=processing_step_args,
    )
    
    # TRAIN STEP
    
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f's3://{default_bucket}/sagemaker-intel-logs-pipeline',
        container_local_output_path='/opt/ml/output/tensorboard'
    )
    
    pt_estimator = PyTorch(
        base_job_name=f"{base_job_name}-training-intel-pipeline",
        source_dir=os.path.join(os.getcwd(), "pipelines/intel/sagemaker-intel"),
        entry_point="train.py",
        role=role,
        py_version="py38",
        framework_version="1.11.0",
        instance_count=1,
        sagemaker_session=pipeline_session,
        instance_type="ml.g4dn.xlarge",
        # instance_type="local",
        tensorboard_output_config=tensorboard_output_config,
        use_spot_instances=True,
        max_wait=5000,
        max_run=5000,
        environment={
            "GIT_USER": "Nikhil Shrimali",
            "GIT_EMAIL": "nshrimali21@gmail.com",
            "MODEL":model_name,
            "Optimizer":optimizer,
            "BatchSize": batchsize,
            "LearningRate": learningrate,
            "EPOCHS":epochs
        }
    )
    
    estimator_step_args = pt_estimator.fit({
        'train': TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
        ),
        'test': TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
        )
    })
    
    step_train = TrainingStep(
        name="TrainIntelClassifier",
        step_args=estimator_step_args,
    )
    
    # EVAL STEP
    
    pytorch_processor = PyTorchProcessor(
        framework_version='1.11.0',
        py_version="py38",
        role=role,
        sagemaker_session=pipeline_session,
        # instance_type='ml.t3.medium',
        # instance_type="ml.c5.xlarge",
        instance_type="ml.m5.4xlarge",
        # instance_type='local',
        instance_count=1,
        base_job_name=f'{base_job_name}-eval-intel-classifier-model',
        # envs={
        #     "MODEL":model_name,
        #     "Optimizer":optimizer,
        #     "BatchSize": batchsize,
        #     "LearningRate": learningrate,
        #     "EPOCHS":epochs
        # }
    )

    
    eval_step_args = pytorch_processor.run(
        code='evaluate.py',
        source_dir=os.path.join(os.getcwd(), "pipelines/intel/sagemaker-intel"),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
    )
    
    evaluation_report = PropertyFile(
        name="IntelClassifierEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateIntelClassifierModel",
        step_args=eval_step_args,
        property_files=[evaluation_report],
    )
    # MODEL REGISTER STEP
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            # s3_uri="s3://sagemaker-ap-south-1-006547668672/eval-flower-classifier-model-2022-12-07-19-40-04-608/output/evaluation/evaluation.json",
            content_type="application/json"
        )
    )
    
    model = PyTorchModel(
        entry_point="infer.py",
        source_dir=os.path.join(os.getcwd(), "pipelines/intel/sagemaker-intel"),
        sagemaker_session=pipeline_session,
        role=role,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        framework_version="1.11.0",
        py_version="py38",
    )

    model_step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m4.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        # approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )
    
    step_register = ModelStep(
        name="RegisterIntelClassifierModel",
        step_args=model_step_args,
    )
    
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value"
        ),
        right=0.6,
    )

    step_cond = ConditionStep(
        name="CheckAccuracyIntelClassifierEvaluation",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,

        parameters=[
            dvc_repo_url,
            dvc_branch,
            input_dataset,
            model_approval_status,
            model_name,
            optimizer,
            batchsize,
            learningrate,
            epochs
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
