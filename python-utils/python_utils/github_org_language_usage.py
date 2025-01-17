from argparse import ArgumentParser

from github_utils import GithubClient

def main(org_name):
    gh = GithubClient()
    org_languages = gh.get_org_languages(org_name)
    
    print(f"Programming languages used in {org_name}:")
    for language, bytes_of_code in org_languages.items():
        print("{}: {}".format(language, bytes_of_code))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("org_name")
    args = parser.parse_args()
    
    main(args.org_name)
