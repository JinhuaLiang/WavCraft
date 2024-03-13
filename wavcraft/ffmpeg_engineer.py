import re
import os
import time
import glob
import pickle
import openai


class FFmpegEngineer:
    OPENAI_KEY = os.environ.get('OPENAI_KEY')

    def __init__(self, use_openai_cache = False,):
        self.use_openai_cache = False
        self.openai_cache = []
        if self.use_openai_cache:
            os.makedirs('cache', exist_ok=True)
            for cache_file in glob.glob('cache/*.pkl'):
                with open(cache_file, 'rb') as file:
                    self.openai_cache.append(pickle.load(file))

        self.history = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            ]

    def complete(self, prompt, model="gpt-4", api_key=""):
        content = self.chat_with_gpt(prompt, model, api_key)
        content = self.extract_ffmpeg_command(self.try_extract_content_from_quotes(content))
        self.execute_code(content)


    def chat_with_gpt(self, prompt, model="gpt-4", api_key=""):
        api_key = self.OPENAI_KEY if not api_key else api_key

        if self.use_openai_cache:
            filtered_object = list(filter(lambda x: x['prompt'] == prompt, self.openai_cache))
            if len(filtered_object) > 0:
                response = filtered_object[0]['response']
                return response
        
        self.history.append(
            {
                "role": "user",
                "content": prompt
            },
            )

        try:
            openai.api_key = api_key
            chat = openai.ChatCompletion.create(
                model=model,  # "gpt-3.5-turbo",
                messages=self.history,
            )
            response = chat['choices'][0]['message']['content']

            self.history.append(
                {
                    "role": "system",
                    "content": response
                },
                )
            
        finally:
            openai.api_key = ''

        if self.use_openai_cache:
            cache_obj = {
                'prompt': prompt,
                'response': response,
            }
            with open(f'cache/{time.time()}.pkl', 'wb') as _openai_cache:
                pickle.dump(cache_obj, _openai_cache)
                self.openai_cache.append(cache_obj)

        return response
    

    def reset(self,):
        self.history = []


    @classmethod
    def _extract_substring_with_quotes(cls, input_string, quotes="'''"):
        pattern = f"{quotes}(.*?){quotes}"
        matches = re.findall(pattern, input_string, re.DOTALL)
        return matches

    @classmethod
    def extract_ffmpeg_command(cls, input_string):
        # Split the string into lines
        lines = input_string.split('\n')

        # Find the index where the 'ffmpeg' command starts
        start_index = next((i for i, line in enumerate(lines) if 'ffmpeg' in line), None)

        # Extract lines from the start of the 'ffmpeg' command till the end or a specific end pattern
        if start_index is not None:
            ffmpeg_lines = lines[start_index:]
            # ffmpeg_lines = ffmpeg_lines[:end_index]
            return '\n'.join(ffmpeg_lines)
        else:
            return ""

    def try_extract_content_from_quotes(self, content):
        if "'''" in content:
            return self._extract_substring_with_quotes(content)[0]
        elif "```" in content:
            return self._extract_substring_with_quotes(content, quotes="```")[0]
        else:
            return content
        
    def execute_code(self, content):
        os.system(content)


"""     Test     """
if __name__ == "__main__":
    import os

    eng = FFmpegEngineer()

    prompt = "Using bash to check the version of ffmpeg in linux."
    model = "gpt-4"
    OPENAI_KEY = "your_key"  # can be set using `${OPENAI_KEY}` from env

    # For examin the func you can use the following lines
    # response = eng.chat_with_gpt(prompt, model, api_key)
    # print(response)
    # code = eng.extract_ffmpeg_command(eng.try_extract_content_from_quotes(response))
    # print(code)
    # eng.execute_code(code)
    # Ones can replace the above lines with one line:
    eng.complete(prompt, model="gpt-4", api_key=OPENAI_KEY)
