import time

import openai
import requests

from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import GetPosts, NewPost
import markdown

from wordpress_xmlrpc.compat import xmlrpc_client
from wordpress_xmlrpc.methods import media

from deep_translator import GoogleTranslator

from multiprocessing import Process

import os


def generate_text(keyword):
    openai.api_key = "OPEN_AI_KEY"
    # models = openai.Model.list()
    # keyword = "retro radio"

    prompt = f"""
    Using markdown formatting, act as an Expert Article Writer and write a fully detailed, long-form, 100% unique, creative, and human-like article of a minimum of 5000 words using headings and sub-headings. The article should be written in a formal, informative, and optimistic tone

    Must Develop a comprehensive "Outline" for a long-form article for the Keyword {keyword}, featuring at least 25 engaging headings and subheadings that are detailed, mutually exclusive, collectively exhaustive, and cover the entire topic. Must use LSI Keywords in these outlines. Must show these "Outlines" in a table.

    Use English for the keyword "{keyword}" and write at least 400–500 words of engaging content under every Heading. This article should show the experience, expertise, authority and trust for the Topic {keyword}. Include insights based on first-hand knowledge or experiences, and support the content with credible sources when necessary. Focus on providing accurate, relevant, and helpful information to readers, showcasing both subject matter expertise and personal experience in the topic {keyword}.


    Try to use contractions, idioms, transitional phrases, interjections, dangling modifiers, and colloquialisms, and avoid repetitive phrases and unnatural sentence structures.

    When you write, you must correctly format the blog post according to proper SEO standards, with as much rich and detailed HTML as possible, for example, lists, bold, italics, quotes from the internet, tables, and external links to high-quality websites such as Wikipedia. Try to ask questions and then immediately give a good and concise answer, to try to achieve the featured snippet on Google.

    The article must include an SEO meta-description right after the title (you must include the {keyword} in the description), an introduction, and a click-worthy short title. Also, use the seed keyword as the first H2. Always use a combination of paragraphs, lists, and tables for a better reader experience.  Use fully detailed paragraphs that engage the reader. Write at least one paragraph with the heading {keyword}. Write down at least six FAQs with answers and a conclusion.

    Note: Don't assign Numbers to Headings. Don't assign numbers to Questions. Don't write Q: before the question (faqs)

    Make sure the article is plagiarism free. Don't forget to use a question mark (?) at the end of questions. Try not to change the original {keyword} while writing the title. Try to use "{keyword}" 2-3 times in the article. Try to include {keyword} in the headings as well. write content that can easily pass the AI detection tools test. Bold all the headings and sub-headings using Markdown formatting.
    """

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                                   messages=[{"role": "user", "content": prompt}])
    print("TEXT GENERATION - SUCCESS")
    return chat_completion.choices[0].message.content


def download_image(keyword):
    photos_json = requests.get(
        "https://api.unsplash.com/search/photos?page=1&query=" + keyword + "&client_id=YOUR_CLIENT_ID").json()

    for i in photos_json["results"]:
        response = requests.get("https://unsplash.com/photos/" + i["id"] + "/download", allow_redirects=True)
        open('unsplash_temp.png', 'wb').write(response.content)
        print("IMAGE DOWNLOAD - SUCCESS")
        break


def upload_image(key_word):
    # Your WordPress username
    username = 'USERNAME'
    password = 'PASSWORD'

    client = Client('https://YOUR_WEBSITE/xmlrpc.php', username, password)

    # set to the path to your file
    filename = 'unsplash_temp.png'

    # prepare metadata
    data = {
        'name': key_word + '.png',
        'type': 'image/jpeg',  # mimetype
    }

    # read the binary file and let the XMLRPC library encode it into base64
    print("ENCODING")
    with open(filename, 'rb') as img:
        data['bits'] = xmlrpc_client.Binary(img.read())

    print("SENDING")
    response = client.call(media.UploadFile(data))

    attachment_id = response['id']
    print("UPLOADING - SUCCESS")
    return attachment_id


def restImgUL():
    url = 'https://YOUR_WEBSITE/wp-json/wp/v2/media'
    data = open("unsplash_temp.png", 'rb').read()
    fileName = os.path.basename("unsplash_temp.png")
    res = requests.post(url='https://YOUR_WEBSITE/wp-json/wp/v2/media',
                        data=data,
                        headers={'Content-Type': 'image/jpg',
                                 'Content-Disposition': 'attachment; filename=%s' % fileName},
                        auth=('WP_USERNAME', 'APPLICATION_PASSWORD'))
    # pp = pprint.PrettyPrinter(indent=4) ## print it pretty.
    # pp.pprint(res.json()) #this is nice when you need it

    newDict = res.json()
    newID = newDict.get('id')
    print("UPLOADING - SUCCESS")
    return newID


def run_keyword(key_word, key_word_category):
    try:
        print(key_word)

        key_word_category_text = key_word_category.split("/")[3].replace("-", " ")

        text_output = markdown.markdown(generate_text(key_word))
        h2_title = text_output.split("\n")[0]
        h2_title = h2_title.replace("<h1>", "")
        h2_title = h2_title.replace("</h1>", "")
        h2_title = h2_title.replace("SEO Meta-Description: ", "")
        seo_meta = text_output.split("\n")[2]

        text_output = text_output.split("\n", 4)[4]

        # The URL for the API endpoint
        url = 'https://YOUR_WEBSITE/wp-json/wp/v2/posts'

        # Your WordPress username
        username = 'USERNAME'
        password = 'PASSWORD'

        wp = Client('https://YOUR_WEBSITE/xmlrpc.php', username, password)

        post = WordPressPost()
        post.title = h2_title
        post.content = text_output
        post.terms_names = {
            'category': [key_word_category_text]
            # Replace with the slugs of the categories you want to assign
        }
        post.custom_fields = []
        post.custom_fields.append({
            'key': '_aioseo_description',
            'value': seo_meta
        })

        translated = GoogleTranslator(source='INPUT_LANGUAGE', target='en').translate(text=key_word)
        download_image(translated)

        post.thumbnail = restImgUL()
        post.post_status = 'publish'

        wp.call(NewPost(post))
        print("PUBLISHED")
        print("------------------")
    except:
        print("NOT PUBLISHED")


with open('keyword_list.txt') as f:
    lines = f.readlines()

with open('keyword_category.txt') as f:
    category_lines = f.readlines()

if __name__ == '__main__':
    for i in range(len(lines)):
        p = Process(target=run_keyword, args=(lines[i], category_lines[i]))
        p.start()