# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

from agenteval.targets import BaseTarget
from agenteval.utils import create_boto3_client, refresh_boto3_client

logger = logging.getLogger(__name__)


class Boto3Target(BaseTarget):
    """A target that can be interfaced with via the `boto3` library.

    Attributes:
        boto3_client (BaseClient): A `boto3` client.
    """

    def __init__(
        self,
        boto3_service_name: str,
        aws_profile: Optional[str] = None,
        aws_region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        max_retry: int = 10,
    ):
        """
        Initialize the AWS target.

        Args:
            boto3_service_name (str): The `boto3` service name (e.g `"bedrock-agent-runtime"`).
            aws_profile (Optional[str]): The AWS profile name.
            aws_region (Optional[str]): The AWS region.
            endpoint_url (Optional[str]): The endpoint URL for the AWS service.
            max_retry (int): The maximum number of retry attempts.
        """

        # Store params to allow client recreation on token expiry
        self._boto3_service_name = boto3_service_name
        self._aws_profile = aws_profile
        self._aws_region = aws_region
        self._endpoint_url = endpoint_url
        self._max_retry = max_retry

        self.boto3_client = create_boto3_client(
            boto3_service_name=boto3_service_name,
            aws_profile=aws_profile,
            aws_region=aws_region,
            endpoint_url=endpoint_url,
            max_retry=max_retry,
        )

    def refresh_boto3_client(self):
        """Recreate the boto3 client with a fresh session.

        This is used to recover from expired SSO/SSO-OIDC tokens by establishing
        a new session that re-reads cached credentials.
        """
        logger.warning(
            "SSO token expired during target invocation, refreshing client..."
        )
        self.boto3_client = refresh_boto3_client(
            boto3_service_name=self._boto3_service_name,
            aws_profile=self._aws_profile,
            aws_region=self._aws_region,
            endpoint_url=self._endpoint_url,
            max_retry=self._max_retry,
        )
