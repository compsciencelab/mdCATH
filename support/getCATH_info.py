import h5py
import json 
import urllib.request
from tqdm import tqdm
from os.path import join as opj

def get_json_from_url(domain, info_error_file):
    base_url = 'https://www.cathdb.info/version/v4_2_0/api/rest/domain_summary/'
    # Open the URL and read the bytes
    try:
        response_bytes = urllib.request.urlopen(opj(base_url, domain)).read()
    except urllib.error.HTTPError as e:
        # if the domain is not found, try with the new version of the API
        try:
            response_bytes = urllib.request.urlopen(opj(base_url.replace('v4_2_0', 'v4_3_0'), domain)).read()
        except urllib.error.HTTPError as e:
            info_error_file.write(f'Error for {domain}: {e}\n')
        return None
    
    # Decode the bytes into a string
    response_str = response_bytes.decode('utf-8')  # Assuming UTF-8 encoding

    # Parse the string into a JSON object or dictionary
    json_data = json.loads(response_str)

    return json_data

if __name__ == '__main__':
    h5_file = '/shared/antoniom/buildCATHDataset/dataloader_h5/mdcath_source.h5'
    info_error_file = open('info_error.txt', 'w')
    info_error_file.write('Errors in getting CATH info\n')
    h5 = h5py.File(h5_file, 'r')
    pdb_list = list(h5.keys())
    h5.close()
    
    info_to_report = ['superfamily_id', 'cath_id', 'domain_id', 'pdb_code']
    cath_info = {}
    for pdb in tqdm(pdb_list, desc='Getting CATH info', total=len(pdb_list)):
        data = get_json_from_url(pdb, info_error_file)
        if data is None:
            continue
        cath_info[pdb] = {}
        for info in info_to_report:
            cath_info[pdb][info] = data['data'][info]
             
    with open('cath_info.json', 'w') as f:
        json.dump(cath_info, f)
    
    print('Done!')