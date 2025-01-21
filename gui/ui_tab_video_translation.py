import os
import time
import traceback
from shortGPT.audio.audio_utils import audioToText, getSpeechBlocks
import gradio as gr

from gui.asset_components import AssetComponentsUtils
from gui.ui_abstract_component import AbstractComponentUI
from gui.ui_components_html import GradioComponentsHTML
from shortGPT.audio.edge_voice_module import EdgeTTSVoiceModule
from shortGPT.audio.eleven_voice_module import ElevenLabsVoiceModule
from shortGPT.config.api_db import ApiKeyManager
from shortGPT.config.languages import (EDGE_TTS_VOICENAME_MAPPING,
                                       ELEVEN_SUPPORTED_LANGUAGES,
                                       LANGUAGE_ACRONYM_MAPPING,
                                          Language)
from shortGPT.engine.multi_language_translation_engine import MultiLanguageTranslationEngine


class VideoTranslationUI(AbstractComponentUI):
    def __init__(self, shortGptUI: gr.Blocks):
        self.shortGptUI = shortGptUI
        self.eleven_language_choices = [lang.value.upper() for lang in ELEVEN_SUPPORTED_LANGUAGES]
        self.embedHTML = '<div style="display: flex; overflow-x: auto; gap: 20px;">'
        self.progress_counter = 0
        self.video_translation_ui = None
        self.speech_blocks = None

    def create_ui(self):
        with gr.Row(visible=False) as video_translation_ui:
            with gr.Column():
                videoType = gr.Radio(["Video file", "Youtube link"], label="Input your video", value="Video file", interactive=True)
                video_path = gr.Video(sources="upload", interactive=True, width=533.33, height=300, visible=True)
                yt_link = gr.Textbox(label="Youtube link (https://youtube.com/xyz): ", interactive=True, visible=False)
                videoType.change(lambda x: (gr.update(visible=x == "Video file"), gr.update(visible=x == "Youtube link")), [videoType], [video_path, yt_link])
                
                transcribeButton = gr.Button("识别音频文本")
                
                with gr.Column(visible=False) as edit_text_ui:
                    text_blocks = gr.Dataframe(
                        headers=["开始时间", "结束时间", "文本内容"],
                        datatype=["number", "number", "str"],
                        col_count=(3, "fixed"),
                        interactive=True,
                        visible=True
                    )
                
                tts_engine = gr.Radio([AssetComponentsUtils.ELEVEN_TTS, AssetComponentsUtils.EDGE_TTS], label="Text to speech engine", value=AssetComponentsUtils.EDGE_TTS, interactive=True)
                with gr.Column(visible=False) as eleven_tts:
                    language_eleven = gr.CheckboxGroup(self.eleven_language_choices, label="Language", value="ENGLISH", interactive=True)
                    voice_eleven = AssetComponentsUtils.voiceChoiceTranslation(provider=AssetComponentsUtils.ELEVEN_TTS)
                with gr.Column(visible=True) as edge_tts:
                    language_edge = gr.CheckboxGroup([lang.value.upper() for lang in Language], label="Language", value="ENGLISH", interactive=True)
               
                tts_engine.change(lambda x: (gr.update(visible=x == AssetComponentsUtils.ELEVEN_TTS), gr.update(visible=x == AssetComponentsUtils.EDGE_TTS)), [tts_engine], [eleven_tts, edge_tts])

                useCaptions = gr.Checkbox(label="Caption video", value=False)
                translateButton = gr.Button("翻译并生成视频", visible=False)
                generation_error = gr.HTML(visible=False)
                video_folder = gr.Button("📁", visible=True)
                output = gr.HTML('<div style="min-height: 80px;"></div>')

            transcribeButton.click(
                self.transcribe_audio,
                inputs=[videoType, video_path, yt_link],
                outputs=[text_blocks, edit_text_ui, translateButton, generation_error]
            )
            transcribeButton.click(
                lambda: gr.update(value="识别音频文本中...", visible=True), 
                inputs=[], 
                outputs=[generation_error]
            )

            video_folder.click(lambda _: AssetComponentsUtils.start_file(os.path.abspath("videos/")))
            translateButton.click(
                self.translate_video,
                inputs=[videoType, yt_link, video_path, tts_engine, language_eleven, language_edge, useCaptions, voice_eleven, text_blocks],
                outputs=[output, video_folder, generation_error]
            )
        self.video_translation_ui = video_translation_ui
        return self.video_translation_ui

    def transcribe_audio(self, videoType, video_path, yt_link, progress=gr.Progress()):
        try:
            if videoType == "Youtube link":
                if not (yt_link.startswith("https://youtube.com/") or yt_link.startswith("https://www.youtube.com/")):
                    raise gr.Error('Invalid YouTube URL')
                audio_path = yt_link
            else:
                if not video_path or not os.path.exists(video_path):
                    raise gr.Error('Invalid video file')
                audio_path = video_path

            progress(0.3, "正在识别音频...")
            whispered = audioToText(audio_path, model_size='base')
            self.speech_blocks = getSpeechBlocks(whispered, silence_time=0.8)
            
            df_data = [[start, end, text] for [start, end], text in self.speech_blocks]
            
            # Save speech blocks persistently
            self._db_speech_blocks = self.speech_blocks.copy()
            
            return (
                df_data,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False)
            )
            
        except Exception as e:
            error_html = GradioComponentsHTML.get_html_error_template().format(
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            return None, gr.update(visible=False), gr.update(visible=False), gr.update(value=error_html, visible=True)

    def translate_video(self, videoType, yt_link, video_path, tts_engine, language_eleven, language_edge, use_captions, voice_eleven, text_blocks, progress=gr.Progress()):
        try:
            # Debugging: Print all input parameters to verify they are passed correctly
            print("Debug: translate_video parameters:")
            print("videoType:", videoType)
            print("yt_link:", yt_link)
            print("video_path:", video_path)
            print("tts_engine:", tts_engine)
            print("language_eleven:", language_eleven)
            print("language_edge:", language_edge)
            print("use_captions:", use_captions)
            print("voice_eleven:", voice_eleven)
            print("text_blocks:", text_blocks)
        
            # Use saved speech blocks if available, otherwise parse from text_blocks
            if hasattr(self, '_db_speech_blocks') and self._db_speech_blocks:
                speech_blocks = self._db_speech_blocks
            else:
                # Debugging: Print text_blocks to verify its content
                print("Debug: text_blocks content before processing:", text_blocks)
        
                # Convert DataFrame to list and skip header
                if hasattr(text_blocks, 'values'):
                    data_rows = text_blocks.values.tolist()
                    print("Debug: data_rows after conversion:", data_rows)
                else:
                    data_rows = text_blocks[1:]
                    print("Debug: data_rows after conversion:", data_rows)
        
                if not data_rows:
                    raise gr.Error('No text blocks found. Please transcribe the video first.')
        
                # 获取视频源
                video_source = yt_link if videoType == "Youtube link" else video_path
        
                speech_blocks = []
                invalid_blocks = []
        
                for row in data_rows:
                    try:
                        # Convert row data to appropriate types
                        start_time = float(row[0]) if row[0] and str(row[0]).strip() != '开始时间' else None
                        end_time = float(row[1]) if row[1] and str(row[1]).strip() != '结束时间' else None
                        text = str(row[2]) if row[2] and str(row[2]).strip() != '文本内容' else None
        
                        # Validate the converted values
                        if start_time is None or end_time is None or text is None:
                            continue
        
                        # Validation
                        if start_time < 0 or end_time < 0:
                            invalid_blocks.append(f"Invalid time values: start={start_time}, end={end_time}")
                            continue
                        if start_time >= end_time:
                            invalid_blocks.append(f"Start time must be less than end time: start={start_time}, end={end_time}")
                            continue
                        if not text.strip():
                            invalid_blocks.append("Empty text content")
                            continue
        
                        speech_blocks.append([[start_time, end_time], text])
                    except (ValueError, IndexError) as e:
                        invalid_blocks.append(f"Error parsing block {row}: {str(e)}")
                        continue
        
                # Provide detailed error message if no valid blocks
                if not speech_blocks:
                    error_msg = 'No valid text blocks found. Please check your edits.\n'
                    if invalid_blocks:
                        error_msg += '\nIssues found:\n- ' + '\n- '.join(invalid_blocks)
                    raise gr.Error(error_msg)
        
            # 获取目标语言列表
            if tts_engine == AssetComponentsUtils.ELEVEN_TTS:
                languages = [Language(lang.lower().capitalize()) for lang in language_eleven]
            elif tts_engine == AssetComponentsUtils.EDGE_TTS:
                languages = [Language(lang.lower().capitalize()) for lang in language_edge]
        
            for i, language in enumerate(languages):
                if tts_engine == AssetComponentsUtils.EDGE_TTS:
                    voice_module = EdgeTTSVoiceModule(EDGE_TTS_VOICENAME_MAPPING[language]['male'])
                if tts_engine == AssetComponentsUtils.ELEVEN_TTS:
                    voice_module = ElevenLabsVoiceModule(ApiKeyManager.get_api_key('ELEVENLABS_API_KEY'), voice_eleven, checkElevenCredits=True)
        
                # 使用我们刚刚解析的 speech_blocks
                content_translation_engine = MultiLanguageTranslationEngine(
                    voiceModule=voice_module,
                    src_url=video_source,
                    target_language=language,
                    use_captions=use_captions,
                    speech_blocks=speech_blocks  # 直接使用刚解析的speech_blocks
                )
        
                num_steps = content_translation_engine.get_total_steps()
                def logger(prog_str):
                    progress(self.progress_counter / (num_steps), f"Translating your video ({i+1}/{len(languages)}) - {prog_str}")
                content_translation_engine.set_logger(logger)
        
                self.progress_counter = 0
        
                for step_num, step_info in content_translation_engine.makeContent():
                    progress(self.progress_counter / (num_steps), f"Translating your video ({i+1}/{len(languages)}) - {step_info}")
                    self.progress_counter += 1
        
                video_path = content_translation_engine.get_video_output_path()
                current_url = self.shortGptUI.share_url+"/" if self.shortGptUI.share else self.shortGptUI.local_url
                file_url_path = f"{current_url}gradio_api/file={video_path}"
                file_name = video_path.split("/")[-1].split("\\")[-1]
        
                self.embedHTML += f'''
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <video width="{500}"  style="max-height: 100%;" controls>
                        <source src="{file_url_path}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <a href="{file_url_path}" download="{file_name}" style="margin-top: 10px;">
                        <button style="font-size: 1em; padding: 10px; border: none; cursor: pointer; color: white; background: #007bff;">Download Video</button>
                    </a>
                </div>'''
                yield "<div>"+self.embedHTML + '</div>', gr.update(visible=True), gr.update(visible=False)
        
        except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            error_name = type(e).__name__.capitalize() + " : " + f"{e.args[0]}"
            print("Error", traceback_str)
            error_html = GradioComponentsHTML.get_html_error_template().format(error_message=error_name, stack_trace=traceback_str)
            return self.embedHTML + '</div>', gr.update(visible=True), gr.update(value=error_html, visible=True)
    def inspect_create_inputs(self, videoType, video_path, yt_link,  tts_engine, language_eleven, language_edge,):
        supported_extensions = ['.mp4', '.avi', '.mov']  # Add more supported video extensions if needed
        print(videoType, video_path, yt_link)
        if videoType == "Youtube link":
            if not yt_link.startswith("https://youtube.com/") and not yt_link.startswith("https://www.youtube.com/"):
                raise gr.Error('Invalid YouTube URL. Please provide a valid URL. Link example: https://www.youtube.com/watch?v=dQw4w9WgXcQ')
        else:
            if not video_path or not os.path.exists(video_path):
                raise gr.Error('You must drag and drop a valid video file.')

            file_ext = os.path.splitext(video_path)[-1].lower()
            if file_ext not in supported_extensions:
                raise gr.Error('Invalid video file. Supported video file extensions are: {}'.format(', '.join(supported_extensions)))
        if tts_engine == AssetComponentsUtils.ELEVEN_TTS:
            if not len(language_eleven) >0:
                raise gr.Error('You must select one or more target languages')
        if tts_engine == AssetComponentsUtils.EDGE_TTS:
            if not len(language_edge) >0:
                raise gr.Error('You must select one or more target languages')
        return gr.update(visible=False)


def update_progress(progress, progress_counter, num_steps, num_shorts, stop_event):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        dynamic = int(3649 * elapsed_time / 600)
        progress(progress_counter / (num_steps * num_shorts), f"Rendering progress - {dynamic}/3649")
        time.sleep(0.1)  # update every 0.1 second
