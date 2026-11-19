"""Read extracted LEVIR-MCI members from the TerraWatch shared Google Drive."""
from urllib.parse import quote

API = 'https://www.googleapis.com/drive/v3/files'
DATASET_FOLDER_ID = '1Ir57tYIXizXGGz1UsOzvuCuRPN7q_Bi8'
CAPTIONS_ID = '1LJpLulDdZODppp2VkgqJFKxbli5SUmKj'
FOLDERS = {
    ('train', 'A'): '1rQpUJT0WD_HJk2yVPzRezgzKiFssVB0G',
    ('train', 'B'): '1E7L9B0kzvzl8dVvJoWjursz2cCVQz5hW',
    ('train', 'label'): '1piC8BRqHJp_5d3XCorfloefvC0O2XkHg',
    ('val', 'A'): '1Uar_vfY4mmDn35L3dU8W5iSAJ3Wx00vQ',
    ('val', 'B'): '1WdRYbs3sbf-31mnlHNbxPelAbLJpDbty',
    ('val', 'label'): '1d6LZrygAT2-DqGw-Fpp_Xxo-1d_SRzJJ',
}


class DriveMembers:
    """Supply the same read(member) contract as zipfile.ZipFile."""

    def __init__(self, session):
        self.session = session
        self.index = {}

    @classmethod
    def connect(cls):
        from google.colab import auth
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        auth.authenticate_user()
        credentials, _ = google.auth.default(
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        session = AuthorizedSession(credentials)
        session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )))
        drive = cls(session)
        drive._get(f'{API}/{DATASET_FOLDER_ID}', {'fields': 'id,name', 'supportsAllDrives': 'true'})
        return drive

    def _get(self, url, params):
        response = self.session.get(url, params=params, timeout=(10, 45))
        if not response.ok:
            try:
                detail = response.json().get('error', {}).get('message', '')
            except ValueError:
                detail = ''
            raise RuntimeError(f'Drive API {response.status_code}: {detail}')
        return response

    def _folder_index(self, folder_id):
        if folder_id in self.index:
            return self.index[folder_id]
        found = {}
        params = {'q': f"'{folder_id}' in parents and trashed = false",
                  'fields': 'nextPageToken,files(id,name)', 'pageSize': 1000,
                  'supportsAllDrives': 'true', 'includeItemsFromAllDrives': 'true'}
        while True:
            page = self._get(API, params).json()
            for file in page.get('files', []):
                name = file['name']
                if name in found and found[name] != file['id']:
                    raise ValueError(f'Duplicate Drive filename {name} in folder {folder_id}')
                found[name] = file['id']
            if not page.get('nextPageToken'):
                break
            params['pageToken'] = page['nextPageToken']
        self.index[folder_id] = found
        return found

    def read(self, member):
        parts = member.split('/')
        if parts == ['LEVIR-MCI-dataset', 'LevirCCcaptions.json']:
            file_id = CAPTIONS_ID
        elif len(parts) == 5 and parts[:2] == ['LEVIR-MCI-dataset', 'images'] and tuple(parts[2:4]) in FOLDERS:
            folder_id = FOLDERS[tuple(parts[2:4])]
            try:
                file_id = self._folder_index(folder_id)[parts[4]]
            except KeyError as error:
                raise FileNotFoundError(member) from error
        else:
            raise FileNotFoundError(member)
        return self._get(f'{API}/{quote(file_id, safe="")}',
                         {'alt': 'media', 'supportsAllDrives': 'true'}).content
