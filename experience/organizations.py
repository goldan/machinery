"""Export organizations names and job title words popularity tables."""
import csv
import sys
from argparse import ArgumentParser
from collections import Counter
from fannypack.csv_utils import CSV_OPTIONS

from machinery.common.atlas import Candidate
from machinery.experience.features import (ProfileJobTitlesWords,
                                           ProfileOrganizationNames)


def get_organization_attr_counter(attribute, normalize_func):
    """Get Counter object for attribute (name/title) of all organizations.

    Find all organizations with existing attribute.
    Normalize attribute values for them.
    Make and return a Counter for those values.
    Used to construct popularity tables for organization names and job titles.

    Args:
        attribute: attribute name to count (name/title).
            If attribute is title, it's divided into words in normalize_title(),
            and words are counted separately.
        normalize_func: function to normalize attribute value.

    Returns:
        collections.Counter object with all counts of the attribute values.
    """
    candidates = Candidate._get_collection().find(
        {'profiles.organizations': {'$exists': True}},
        {'profiles.organizations.%s' % attribute: 1})
    counter = Counter()
    i = 1
    for candidate in candidates:
        print "\r%d" % i,
        sys.stdout.flush()
        i += 1
        for profile in candidate['profiles']:
            for organization in profile.get('organizations', []):
                value = organization.get(attribute)
                if value:
                    values = normalize_func(organization[attribute])
                    values = set([values]) if not isinstance(values, set) else values
                    for value in values:
                        counter[value] += 1
    return counter


def export_organizations_popularity(attribute, normalize_func, output_filename):
    """Export a csv table with organization attribute values and their frequency in jobs.

    Args:
        attribute: attribute name to count (name/title).
        normalize_func: function to normalize attribute value.
        output_filename: name of the csv file to output information to.
    """
    with open(output_filename, "wb") as fout:
        writer = csv.writer(fout, **CSV_OPTIONS)
        counter = get_organization_attr_counter(attribute, normalize_func)
        counter_sorted = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
        for name, count in counter_sorted:
            writer.writerow((name.encode('utf8'), count))


def main():
    """Function that is executed when a file is called from CLI."""
    parser = ArgumentParser()
    parser.add_argument('output_filename', type=str)
    parser.add_argument('mode', type=str)
    options = parser.parse_args()
    if options.mode == "name":
        export_organizations_popularity(
            "name", ProfileOrganizationNames.normalize_name, options.output_filename)
    elif options.mode == "title":
        export_organizations_popularity(
            "title", ProfileJobTitlesWords.normalize_title, options.output_filename)
    else:
        raise Exception("Specify mode: name or title")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
