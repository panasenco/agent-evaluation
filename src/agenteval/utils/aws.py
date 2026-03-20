# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import boto3
from botocore.client import BaseClient
from botocore.config import Config

_RETRY_MODE = "adaptive"

logger = logging.getLogger(__name__)


def create_boto3_client(
    boto3_service_name: str,
    aws_profile: Optional[str],
    aws_region: Optional[str],
    endpoint_url: Optional[str],
    max_retry: int,
) -> BaseClient:
    """Create a `boto3` client.

    Args:
        boto3_service_name (str): The `boto3` service name (e.g `"bedrock-runtime"`).
        aws_profile (Optional[str]): The AWS profile name.
        aws_region (Optional[str]): The AWS region.
        endpoint_url (Optional[str]): The endpoint URL for the AWS service.
        max_retry (int): The maximum number of retry attempts.

    Returns:
        BaseClient
    """

    config = Config(retries={"max_attempts": max_retry, "mode": _RETRY_MODE})

    session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
    return session.client(boto3_service_name, endpoint_url=endpoint_url, config=config)


def refresh_boto3_client(
    boto3_service_name: str,
    aws_profile: Optional[str],
    aws_region: Optional[str],
    endpoint_url: Optional[str],
    max_retry: int,
) -> BaseClient:
    """Create a fresh `boto3` client by establishing a new session.

    This is used to recover from expired SSO tokens by forcing a new session
    which re-reads cached SSO credentials.

    Args:
        boto3_service_name (str): The `boto3` service name (e.g `"bedrock-runtime"`).
        aws_profile (Optional[str]): The AWS profile name.
        aws_region (Optional[str]): The AWS region.
        endpoint_url (Optional[str]): The endpoint URL for the AWS service.
        max_retry (int): The maximum number of retry attempts.

    Returns:
        BaseClient
    """
    logger.info("Refreshing boto3 client due to expired credentials/token")
    return create_boto3_client(
        boto3_service_name=boto3_service_name,
        aws_profile=aws_profile,
        aws_region=aws_region,
        endpoint_url=endpoint_url,
        max_retry=max_retry,
    )
