"""Import this module before importing any atlas models, e.g. Candidate."""
import os

import django

os.environ["DJANGO_SETTINGS_MODULE"] = "atlas_project.global.settings"
django.setup()

from atlas_project.candidate.models import Candidate, Organization
from atlas_project.candidate.profile_defs import (GENDER_MALE,
                                                  PROFILE_TYPES_BY_TYPE_ID)
