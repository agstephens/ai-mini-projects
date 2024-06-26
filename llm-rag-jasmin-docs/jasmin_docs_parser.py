import os
import requests

import git
import pandas as pd


GIT_REPO_URL = "https://github.com/cedadev/jasmin-help-hugo-hinode"
GIT_REPO_PATH = "./jasmin-help-hugo-hinode"
CSV_PATH = "./jasmin_docs.csv"


def clone_repo(url, path):
    if not os.path.isdir(path):
        print(f"Cloning {url} to {path}.")
        git.Repo.clone_from(url, path)
    else:
        print(f"Repo already exists at {path}.")


def check_url(url):
    response = requests.get(url)
    response.raise_for_status()


def parse_md_file(file_path, docs_url="https://help.jasmin.ac.uk/docs"):
    """
    Parse a markdown file and return a list of sections.
    
    Args:
        file_path (str): The path to the markdown file.
        docs_url (str): The base URL for the documentation site.
        
    Returns:
        list: A list of sections, each represented as a list with three elements:
            - The section title.
            - The section content.
            - The section URL.
    """
    print(f"Parsing {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    page_slug_search = [line.split(":")[-1].strip() for line in lines[:20] 
                       if line.startswith("slug:")]
    if page_slug_search:
        page_slug = page_slug_search[0]
    else:
        page_slug = file_path.split(os.path.sep)[-1][:-3]

    url_parts = file_path.split(os.path.sep)[3:]
    url_base = docs_url + "/" + "/".join(url_parts)[:-3]

    if url_base.endswith("_index"):
        url_base = url_base[:-6]
    else:
        url_base = "/".join(url_base.split("/")[:-1] + [page_slug])

    check_url(url_base)

    try:
        section_separator = sorted(set(
                [line.strip().split()[0] for line in lines if line.startswith("#")]
            ))[0]
    except IndexError:
        print(f"No sections found in {file_path}.")
        return []

    sections = []
    current_section = None

    in_code_block = False
    triple_dash_count = 0

    for line in lines:
        if line.strip() == "---":
            triple_dash_count += 1
            continue

        if triple_dash_count == 1 and line.startswith("title:"):
            page_title = line.split(":")[1].strip()
            continue

        if triple_dash_count < 2:
            continue

        if not line.strip():
            continue

        if line.startswith("```"):
            in_code_block = not in_code_block
            continue

        if line.startswith(section_separator) and not in_code_block:
            if current_section:
                sections.append(current_section)

            title = line.strip("#").strip()
            slug = title.lower().replace(" ", "-")
            current_section = [line.strip("#").strip(), "", f"{url_base}#{slug}"]
        else:
            if not current_section:
                current_section = [page_title, "", url_base]
            current_section[1] += line

    if not sections or sections[-1] != current_section:
        sections.append(current_section)

    return sections


def parse_repo(git_repo_url=GIT_REPO_URL, 
               repo_path=GIT_REPO_PATH, 
               csv_path=CSV_PATH, max_docs=5000):

    exclude_files = ["centos7-sci-login-xfer-servers.md"]

    if os.path.isfile(csv_path):
        print(f"Loading {csv_path}.")
        return pd.read_csv(csv_path)

    clone_repo(git_repo_url, repo_path)

    base_path = os.path.join(repo_path, "content", "docs")
    sections = []

    count = 0
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if count > max_docs: 
                break

            if file.endswith(".md") and file not in exclude_files:
                new_sections = parse_md_file(os.path.join(root, file))
                if new_sections:
                    sections.extend(new_sections)
                    count += 1


    print(f"Parsed {count} files.")
    df = pd.DataFrame(sections, columns=["title", "contents", "page_url"])

    df["char_count"] = df["contents"].apply(len)
    df["word_count"] = df["contents"].apply(lambda x: len(x.split()))
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":

    df = parse_repo()
    print(df.shape)
    print(df.head(3))

