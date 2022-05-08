from ast import keyword
from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for, session, current_app, Flask, send_file
from transformers import T5ForConditionalGeneration,T5Tokenizer
import nltk
nltk.download('stopwords')
nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
nltk.download('punkt')
nltk.download('wordnet')
from Questgen import main
from Questgen.mcq import mcq
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import random
import json
import requests
import re
import string
import itertools
import traceback
from flashtext import KeywordProcessor
import textwrap
from nltk.corpus import stopwords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


views = Blueprint('views', __name__)

global MainQGen
MainQGen = None

global T5BaseTokenizer
T5BaseTokenizer = None

global T5BaseModel
T5BaseModel = None

global BoolQ
BoolQ = None

global studyguide_finaloutput
studyguide_finaloutput = None

global test_finaoutput
test_finaloutput = None


@views.route('/')
def home():
    return redirect(url_for('views.About'))



def GetRemainder(num1, num2):
    while(num1 >= num2):
        num1 -= num2
    return num1

def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def GetKeywords(keywordlist , amount):
    keywords = []
    i = 0
    while len(keywords) != amount and i < 300:
        i += 1
        keyword = random.choice(keywordlist)
        if keyword not in keywords:
            keywords.append(keyword)
    return keywords

def EditKeywordList(keywordlist, total_amount):
    newlist = []
    for wordkey in keywordlist:
        for _ in range(4):
            newlist.append(wordkey)

    return newlist

def GetListNoDuplicates(original_list):
    newlist = []
    for item in original_list:
        if not item in newlist:
            newlist.append(item)
    return newlist

def postprocesstext (content):
    final=""
    for sent in sent_tokenize(content):
      sent = sent.capitalize()
      final = final +" "+sent
    return final

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)
    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=False)
        keyword_sentences[key] = values
    return keyword_sentences

def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary



@views.route('/Study-Guide', methods=['GET', 'POST'])
def Study_Guide():
    if not 'mcq_amount' in session:
        session["mcq_amount"] = "5"
        session["shortq_amount"] = "5"
        session["yn_amount"] = "5"
        session["match_amount"] = "0"
        session["keyword_amount"] = "5"
    q_amounts = [session["mcq_amount"], session["shortq_amount"], session["yn_amount"], session["match_amount"], session["keyword_amount"]]
    global T5BaseModel
    global T5BaseTokenizer
    global MainQGen
    global BoolQ
    global studyguide_finaloutput
    if request.method == 'POST':
        print("YOOO")
        if request.form.get('submit_study_guide') == 'Generate study guide':
            print("Submit")
            oldcontext = request.form.get('context')
            contextsplit = oldcontext.split('?break')
            for contxt in contextsplit:
                contxt = contxt.strip()
            total_amount = session["total_amount"]
            if T5BaseModel == None:
                T5BaseModel = T5ForConditionalGeneration.from_pretrained('t5-base')
            if T5BaseTokenizer == None:
                T5BaseTokenizer = T5Tokenizer.from_pretrained('t5-base')
            if MainQGen == None:
                MainQGen = main.QGen()
            if BoolQ == None and int(q_amounts[2])>0:
                BoolQ = main.BoolQGen()
            summary_model = T5BaseModel
            summary_tokenizer = T5BaseTokenizer
            summary_model = summary_model.to(device)
            summarys = []
            mcq_qs = []
            yn_qs = []
            shortq_qs = []
            keyword_notqs = []
            for context in contextsplit:
                keywords = MainQGen.GetKeywords(context)
                keywords = EditKeywordList(keywords, round(total_amount/len(contextsplit)))
                summarizedtext = summarizer(context,summary_model,summary_tokenizer)
                summarys.append(summarizedtext)
                if int(session['mcq_amount']) > 0:
                    mcq_keywords = GetKeywords(keywords,int(session['mcq_amount']))
                    print('MCQ KEYWORDS:')
                    print(mcq_keywords)
                    for mcq_keyword in mcq_keywords:
                        keywords.remove(mcq_keyword)
                    mcq_questions = MainQGen.predict_mcq_keywords(context,mcq_keywords)
                    print('MCQ QUESTIONS:')
                    for mcq in mcq_questions['questions']:
                        print(mcq['question_statement'])
                else:
                    mcq_questions = None
                

                if int(session['shortq_amount']) > 0:
                    shortq_keywords = GetKeywords(keywords,int(session['shortq_amount']))
                    for shortq_keyword in shortq_keywords:
                        keywords.remove(shortq_keyword)
                    shortq_quetions = MainQGen.predict_shortq_keywords(context,shortq_keywords)
                else:
                    shortq_quetions = None

                if int(session['yn_amount']) > 0:
                    yn_keywords = GetKeywords(keywords,int(session['yn_amount']))
                    for yn_keyword in yn_keywords:
                        keywords.remove(yn_keyword)
                    yn_quetions = BoolQ.predict_boolq_keywords(context,int(session['yn_amount']))
                else:
                    yn_quetions = None

                if int(session['keyword_amount']) > 0:
                    cut_amount = int(session['keyword_amount'])
                    normal_keywords = MainQGen.GetKeywords(context)
                    if cut_amount >= len(normal_keywords):
                        cut_amount = len(normal_keywords) - 1
                    normal_keywords = normal_keywords[:cut_amount]


                for questyon in mcq_questions['questions']:
                    mcq_qs.append(questyon)

                for questyon in yn_quetions['Boolean Questions']:
                    yn_qs.append(questyon)
                
                for questyon in shortq_quetions['questions']:
                    shortq_qs.append(questyon)
                
                for questyon in normal_keywords:
                    keyword_notqs.append(questyon)

            

            studyguide_finaloutput = {}

            studyguide_finaloutput['summarys'] = summarys
            studyguide_finaloutput['shortq_questions'] = shortq_qs
            studyguide_finaloutput['yn_questions'] = yn_qs
            studyguide_finaloutput['mcq_questions'] = mcq_qs
            studyguide_finaloutput['keywords'] = keyword_notqs
            print(keyword_notqs)

            print(yn_qs)

            

            return render_template('studyguide.html', qamounts=q_amounts, final_output=studyguide_finaloutput)
        elif request.form.get('savesettings') == 'Save':
            session["mcq_amount"] = request.form.get("mcq_amount")
            session["shortq_amount"] = request.form.get("shortq_amount")
            session["yn_amount"] = request.form.get("yn_amount")
            session["keyword_amount"] = request.form.get("keyword_amount")
            q_amounts = [session["mcq_amount"], session["shortq_amount"], session["yn_amount"], session["match_amount"], session["keyword_amount"]]
            session["total_amount"] = int(q_amounts[0])+int(q_amounts[1])+int(q_amounts[2])+int(q_amounts[3])+int(q_amounts[4])
            return render_template('studyguide.html', qamounts=q_amounts, final_output=studyguide_finaloutput)
        else:
            return render_template('studyguide.html', qamounts=q_amounts, final_output=studyguide_finaloutput)
    elif request.method == 'GET':
        return render_template('studyguide.html', qamounts=q_amounts, final_output=studyguide_finaloutput)
    
    return render_template("studyguide.html", qamounts=q_amounts, final_output=studyguide_finaloutput)




@views.route('/Test', methods=['GET', 'POST'])
def Test():
    if not 'mcq_amount' in session:
        session["mcq_amount"] = "5"
        session["shortq_amount"] = "5"
        session["yn_amount"] = "5"
        session["match_amount"] = "0"
        session["keyword_amount"] = "5"
    q_amounts = [session["mcq_amount"], session["shortq_amount"], session["yn_amount"], session["match_amount"], session["keyword_amount"]]
    global T5BaseModel
    global T5BaseTokenizer
    global MainQGen
    global BoolQ
    global test_finaloutput
    if request.method == 'POST':
        print("YOOO")
        if request.form.get('submit_study_guide') == 'Generate test':
            print("Submit")
            oldcontext = request.form.get('context')
            contextsplit = oldcontext.split('?break')
            for contxt in contextsplit:
                contxt = contxt.strip()
            total_amount = session["total_amount"]
            if T5BaseModel == None:
                T5BaseModel = T5ForConditionalGeneration.from_pretrained('t5-base')
            if T5BaseTokenizer == None:
                T5BaseTokenizer = T5Tokenizer.from_pretrained('t5-base')
            if MainQGen == None:
                MainQGen = main.QGen()
            if BoolQ == None and int(q_amounts[2])>0:
                BoolQ = main.BoolQGen()
            summary_model = T5BaseModel
            summary_tokenizer = T5BaseTokenizer
            summary_model = summary_model.to(device)
            summarys = []
            mcq_qs = []
            yn_qs = []
            shortq_qs = []
            keyword_notqs = []
            for context in contextsplit:
                keywords = MainQGen.GetKeywords(context)
                keywords = EditKeywordList(keywords, round(total_amount/len(contextsplit)))
                summarizedtext = summarizer(context,summary_model,summary_tokenizer)
                summarys.append(summarizedtext)
                if int(session['mcq_amount']) > 0:
                    mcq_keywords = GetKeywords(keywords,int(session['mcq_amount']))
                    print('MCQ KEYWORDS:')
                    print(mcq_keywords)
                    for mcq_keyword in mcq_keywords:
                        keywords.remove(mcq_keyword)
                    mcq_questions = MainQGen.predict_mcq_keywords(context,mcq_keywords)
                    print('MCQ QUESTIONS:')
                    for mcq in mcq_questions['questions']:
                        print(mcq['question_statement'])
                else:
                    mcq_questions = None
                

                if int(session['shortq_amount']) > 0:
                    shortq_keywords = GetKeywords(keywords,int(session['shortq_amount']))
                    for shortq_keyword in shortq_keywords:
                        keywords.remove(shortq_keyword)
                    shortq_quetions = MainQGen.predict_shortq_keywords(context,shortq_keywords)
                else:
                    shortq_quetions = None

                if int(session['yn_amount']) > 0:
                    yn_keywords = GetKeywords(keywords,int(session['yn_amount']))
                    for yn_keyword in yn_keywords:
                        keywords.remove(yn_keyword)
                    yn_quetions = BoolQ.predict_boolq_keywords(context,int(session['yn_amount']))
                else:
                    yn_quetions = None

                if int(session['keyword_amount']) > 0:
                    cut_amount = int(session['keyword_amount'])
                    normal_keywords = MainQGen.GetKeywords(context)
                    if cut_amount >= len(normal_keywords):
                        cut_amount = len(normal_keywords) - 1
                    normal_keywords = normal_keywords[:cut_amount]


                for questyon in mcq_questions['questions']:
                    mcq_qs.append(questyon)

                for questyon in yn_quetions['Boolean Questions']:
                    yn_qs.append(questyon)
                
                for questyon in shortq_quetions['questions']:
                    shortq_qs.append(questyon)
                
                for questyon in normal_keywords:
                    keyword_notqs.append(questyon)

            

            test_finaloutput = {}

            test_finaloutput['summarys'] = summarys
            test_finaloutput['shortq_questions'] = shortq_qs
            test_finaloutput['yn_questions'] = yn_qs
            test_finaloutput['mcq_questions'] = mcq_qs
            test_finaloutput['keywords'] = keyword_notqs
            print(keyword_notqs)

            print(yn_qs)

            

            return render_template('test.html', qamounts=q_amounts, final_output=test_finaloutput)
        elif request.form.get('savesettings') == 'Save':
            session["mcq_amount"] = request.form.get("mcq_amount")
            session["shortq_amount"] = request.form.get("shortq_amount")
            session["yn_amount"] = request.form.get("yn_amount")
            session["keyword_amount"] = request.form.get("keyword_amount")
            q_amounts = [session["mcq_amount"], session["shortq_amount"], session["yn_amount"], session["match_amount"], session["keyword_amount"]]
            session["total_amount"] = int(q_amounts[0])+int(q_amounts[1])+int(q_amounts[2])+int(q_amounts[3])+int(q_amounts[4])
            return render_template('test.html', qamounts=q_amounts, final_output=test_finaloutput)
        else:
            return render_template('test.html', qamounts=q_amounts, final_output=test_finaloutput)
    elif request.method == 'GET':
        return render_template('test.html', qamounts=q_amounts, final_output=test_finaloutput)
    
    return render_template("test.html", qamounts=q_amounts, final_output=test_finaloutput)


@views.route('/About')
def About():
    return render_template('about.html')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'Machine_Learning'
app.secret_key = 'Machine_Learning'
app.register_blueprint(views, url_prefix='/')






#{%  if qamounts[3] != 0 %}
#        <b>Match the following:</b><br>
#        <div style="width:50%;">
#            {% for match1 in final_output['matches1'] %}
#            <p>___ {{match1}}</p>
#            {% endfor %}
#        </div>
#        <div style="width:50%;">
#            <ul>
#                {% for match2 in final_output['matches2'] %}
#                <li>{{match2}}</p></li>
#                {% endfor %}
#            </ul>
#        </div>
#        {% endif %}