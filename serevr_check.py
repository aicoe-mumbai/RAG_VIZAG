import requests
import json

# Set up the URL, headers, and data for the POST request
url = 'http://172.16.34.235:8080/v1/chat/completions'  # Replace with your correct URL
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer <your_api_token>'  # Optional if your API requires a token
}

data = {
    'model': 'tgi',
    'messages': [
        {
            'role': 'system',
            'content': '\n an AI assistant designed to assist users by providing simple and clear answers to their questions.\nINSTRUCTIONS:\n- Avoid repeating the same phrase or sentence multiple times.\n- Context is generated from database so user is not aware about context, so understand the user question and respond to it.\n\nProvide a concise response unless the user requests more details.'
        },
        {
            'role': 'user',
            'content': 'Refer to the Context scrapped from Vector Database File: /home/qa-prod/Desktop/QC_Documents/ASME SEC V B SE-1030.pdf\nPage: 6\nText: SE-1030 2004 SECTION V APPENDIXES (Nonmandatory Information) X1. RADIOGRAPHIC STANDARD SHOOTING SKETCH (RSS) X1.1 The radiographic standard shooting sketch (RSS) provides the radiographic operator and the radio- graphic interpreter with pertinent information regarding the examination of a casting. The RSS is designed to standardize radiographic methodologies associated with casting examination; it may also provide a means of a purchaser and supplier agreement, prior to initiation of the examination on a production basis. The use of a RSS is advantageous due to the many conﬁgurations associated with castings and the corresponding variations in tech- niques for inspection of any particular one. The RSS provides a map of location marker placement, directions for source and ﬁlm arrangement, and instructions for all other parameters associated with radiography of a casting. This information serves to provide the most efﬁcient method for controlling the quality and consistency of the resultant radiographic representations. X1.2 The RSS usually consists of an instruction sheet and sketch(es) of the casting: the instruction sheet speci-\n---\nFile: /home/qa-prod/Desktop/QC_Documents/ASME SEC V B SE-1030.pdf\nPage: 4\nText: 9.8 Location Markers — The radiographic image of the location markers for the coordination of the casting with the ﬁlm shall appear on the ﬁlm, without interfering with the interpretation, in such an arrangement that it is evident that the required coverage was obtained. These marker positions shall be marked on the casting and the position of the markers shall be maintained on the part during the complete radiographic cycle. The RSS shall show all marker locations. 9.9 Radiographic Identiﬁcation — A system of posi- tive identiﬁcation of the ﬁlm shall be provided. As a minimum, the following shall appear on the radiograph: the name or symbol of the examining laboratory, the date, the casting identiﬁcation number, and whether it is an original or subsequent exposure. 9.10 Subsequent Exposure Identiﬁcation — All repair radiographs after the original (initial) shall have an exami- nation status designation that indicates the reason. Subse- quent radiographs made by reason of a repaired area shall be identiﬁed with the letter “R” followed by the respective repair cycle (that is, R-1 for the ﬁrst repair, R-2 for the second repair, etc.). Subsequent radiographs that are\n---\nFile: /home/qa-prod/Desktop/QC_Documents/ASME SEC V B SE-1030.pdf\nPage: 4\nText: cent to the penetrameter (IQI) through the body of the shim or separate block shall not exceed the density mea- sured in the area of interest by more than 15%. The density may be lighter than the area of interest density, provided acceptable quality level is obtained and the den- sity requirements of 8.6 are met. 282 9.7.6.3 The shim or separate block shall be placed at the corner of the ﬁlm holder or close to that part of the area of interest that is furthest from the central beam. This is the worst case position from a beam angle standpoint that a discontinuity would be in. 9.7.6.4 The shim or separate block dimensions shall exceed the penetrameter (IQI) dimensions such that the outline of at least three sides of the penetrameter (IQI) image shall be visible on the radiograph. 9.7.7 Film Side Penetrameter (IQI) — In the case where the penetrameter (IQI) cannot be physically placed on the source side and the use of a separate block tech- nique is not practical, penetrameters (IQI’s) placed on the ﬁlm side may be used. The applicable job order or contract shall dictate the requirements for ﬁlm side radio- graphic quality level (see 8.4). and answer for user question hi'
        }
    ],
    'stream': True,
    'max_tokens': 1500
}

# Making the POST request with error handling
response = requests.post(url, headers=headers, data=json.dumps(data), proxies={"http": None, "https": None}, stream=True)

print("Response Status Code:", response.status_code)

# Checking if the response status is OK (200)
if response.status_code == 200:
    for chunk in response.iter_lines():
        if chunk:
            try:
                # Decode the byte string (if chunk is in bytes)
                decoded_chunk = chunk.decode('utf-8')

                # Remove the "data:" prefix from the chunk before parsing JSON
                if decoded_chunk.startswith("data:"):
                    decoded_chunk = decoded_chunk[5:].strip()  # Remove 'data:' and any leading/trailing whitespaces
                
                # Check if chunk is '[DONE]' and skip it
                if decoded_chunk == "[DONE]":
                    break  # End the loop if '[DONE]' is encountered
                
                # Now parse the cleaned-up chunk as JSON
                chunk_data = json.loads(decoded_chunk)
                content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                
                if content:  # Only print non-empty content
                    print(content)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Raw chunk (decoded): {decoded_chunk}")
            except Exception as e:
                print(f"Error processing chunk: {e}")
else:
    print(f"Error: {response.status_code}")