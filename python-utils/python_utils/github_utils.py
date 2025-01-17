import json
import os

import requests
from dotenv import load_dotenv

class GithubClient:
    def __init__(self):
        self.base_url = "https://api.github.com"
        
        load_dotenv()
        github_token = os.getenv('GITHUB_TOKEN')
        assert github_token, "Environment variable GITHUB_TOKEN not loaded properly"
        
        self.headers = {"Authorization": "token {}".format(github_token)}
    
    def api_request(self, url, request=requests.get):
        """API request - using class attribute "headers", error handling, and response parsed with json.loads
        Args:
            url (str): _description_
            request (optional): API method from requests (e.g. requests.get, requests.post). Defaults to requests.get.
        Returns:
            dict: API response.content parsed using json.loads
        """
        response = request(url, headers=self.headers)
        response.raise_for_status()
        parsed_response = json.loads(response.content) if response.content else None
        
        return parsed_response
    
    def get_org_repos(self, org_name):
        """Gets all the repositories in a GitHub organization.
        Args:
            org_name: The name of the GitHub organization.
        Returns:
            list: each element is a repository dictionary (from GitHub API response: /orgs/{org_name}/repos)
        """
        
        url = f"{self.base_url}/orgs/{org_name}/repos"
        repos = self.api_request(url=url)
        return repos
    
    def get_repo_languages(self, repo):
        """Gets the programming languages used in a GitHub repository.
        Args:
            repo: A GitHub repository dictionary (taken from GitHub API response)
        
        Returns:
            dict of {str: int}:
                key: programming language name
                value: number of bytes of code written in the language
        """
        
        repo_owner_login = repo.get("owner").get("login")
        repo_name = repo.get("name")
        url = f"https://api.github.com/repos/{repo_owner_login}/{repo_name}/languages"
        languages = self.api_request(url=url)
        return languages
    
    def get_org_languages(self, org_name):
        """Gets all the programming languages used in a GitHub organization.
        Args:
            org_name: The name of the GitHub organization
        
        Returns:
            dict of {str: int}:
                key: programming language name
                value: total number of bytes of code written in the language across all repositories in the organization
        """
        
        org_languages = {}
        repos = self.get_org_repos(org_name)
        for repo in repos:
            repo_languages = self.get_repo_languages(repo)
            for language, bytes_of_code in repo_languages.items():
                if language not in org_languages:
                    org_languages[language] = 0
                org_languages[language] += bytes_of_code
        
        org_languages_sortdesc = {
            k: v for k, v in sorted(
                org_languages.items(),
                key=lambda item: item[1],
                reverse=True)
        }
        
        return org_languages_sortdesc
