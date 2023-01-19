from pydoc import plain
from unittest import result
from flask import Flask, flash, render_template, request
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import binascii
import os, re
from os import listdir
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from werkzeug.utils import secure_filename
import subprocess, sys
import secrets
import math
from math import log10, sqrt
import pickle
from ThreadedFileLoader.ThreadedFileLoader import *
# from app import nonce

app = Flask(__name__)

secret = secrets.token_urlsafe(32)

encode_folder = "Encoded_image_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M"))
app.secret_key = "cariocoders-ednalan"
UPLOAD_FOLDER = os.getcwd() + '/static/uploads/'
RESULT_FOLDER = os.getcwd() + '/static/result/'
DECODE_FOLDER = os.getcwd() + '/static/decode/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER 
app.config['DECODE_FOLDER'] = DECODE_FOLDER
app.config['MAX_CONTENT_LENGHT'] = 16 * 1024 * 1024
MAX_COLOR_VALUE = 256
MAX_BIT_VALUE = 8

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# methods
# AES
# nonce = Nonce

def go_encrypt(arr_data,arr_key):
    cipher = AES.new(arr_key, AES.MODE_EAX) 
    nonce = cipher.nonce
    print("nonce : ", nonce)
    data = cipher.encrypt(arr_data)
    # cipher = [nonce, cipherText, tag]
    d = binascii.hexlify(data)
    cipherText = d.decode("utf-8")
    print("data ciphertext : ", cipherText)
    return nonce, cipherText

def go_decrypt(arr_key, nonce, decrypt):
    decrypted = binascii.unhexlify(decrypt)
    print("devrypt bytes = ", decrypted)
    cipher = AES.new(arr_key, AES.MODE_EAX, nonce)
    p = cipher.decrypt(decrypted)
    plaintext = p.decode("utf-8")
    print("plaintext = ", plaintext)
    return plaintext

# PSNR & MSE
def go_psnr(stego, carrier_img):
    cover_img = carrier_img
    stego_img = stego
    dimensiCover = np.shape(cover_img)
    dimensiStego = np.shape(stego_img)
    print("dimensi gambar cover = ",dimensiCover[0],"*",dimensiCover[1]," = ",dimensiCover[0]*dimensiCover[1])
    print("dimensi gambar stego = ",dimensiStego[0],"*",dimensiStego[1]," = ",dimensiStego[0]*dimensiStego[1])
    # [0] = height, [1] = width, [2] = color channel
    if (dimensiCover[0]*dimensiCover[1]) != (dimensiStego[0]*dimensiStego[1]):
        mse = (dimensiCover[0]*dimensiCover[1])-(dimensiStego[0]*dimensiStego[1])
        # psnr = "Beda Dimensi"
    else:
        mse = np.mean(np.subtract(cover_img.astype(int),stego_img.astype(int)) ** 2)
        
    if mse == 0:
        psnr = 100
    else:
        PIXEL_MAX = 255.0
        psnr = round(20 * math.log10(PIXEL_MAX / math.sqrt(mse)), 2)  #  ,2)
    
    return mse,psnr

def PSNR(compressed, original):
    cover_img = original
    # print("cover : ", cover_img)
    stego_img = compressed
    # print("stego : ",stego_img)
    dimensiCover = np.shape(cover_img)
    dimensiStego = np.shape(stego_img)
    print("dimensi gambar cover = ",dimensiCover[0],"*",dimensiCover[1]," = ",dimensiCover[0]*dimensiCover[1])
    print("dimensi gambar stego = ",dimensiStego[0],"*",dimensiStego[1]," = ",dimensiStego[0]*dimensiStego[1])

    # mse = np.mean((cover_img - stego_img) ** 2)
    if (dimensiCover[0]*dimensiCover[1]) != (dimensiStego[0]*dimensiStego[1]):
        mse = (dimensiCover[0]*dimensiCover[1])-(dimensiStego[0]*dimensiStego[1])
        # psnr = "Beda Dimensi"
    else:
        mse = np.mean((cover_img - stego_img) ** 2)

    # if(mse == 0): # MSE is zero means no noise is present in the signal .
    #       # Therefore PSNR have no importance.
    #   return 100
    # max_pixel = 255.0
    # psnr = 20 * log10(max_pixel / sqrt(mse))
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
    # print("mse psnr : ",mse,psnr)
    return mse, psnr

def loadImages():
    # return array of images
    path = "/xampp/htdocs/stego/static/uploads"

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = Image.open(path + image)
        loadedImages.append(img)

    return loadedImages
  

def encode_text(cipherText, carrier_img, filename):
    img = carrier_img
    # img = cv2.imread(image_name)

    data = cipherText
    #buat hasil stego
    file_name = "encoded_" + filename
    #buat ngitung MSE & PSNR
    file = filename + ".jpg"

    encoded_data = hideData(img, data)
    # encoded_data = lsb_encode(img, data)
    stego = cv2.imwrite(RESULT_FOLDER + file_name, encoded_data)
    stego2 = cv2.imwrite(RESULT_FOLDER + file, encoded_data)
    stego = cv2.imread(RESULT_FOLDER + file_name, 1)
    stego2 = cv2.imread(RESULT_FOLDER + file, 1)  
    # print(stegoLSB)
    return stego, stego2, file_name

def data2binary(data):
    if type(data) == str:
        return ''.join([format(ord(i),"08b") for i in data])
    elif type(data) == bytes or type(data) == np.ndarray:
      return [format(i,"08b") for i in data]

def hideData(img, text): 
    text += "#####"

    data_index = 0
    binary_data = data2binary(text)
     
    # binary_data = text
    data_length = len(binary_data)
    
    for values in img:
        for pixel in values:
            
            r,g,b = data2binary(pixel)
    
            if data_index < data_length:
                pixel[0] = int(r[:-1] + binary_data[data_index])
                data_index += 1
            if data_index < data_length:
                pixel[1] = int(g[:-1] + binary_data[data_index])
                data_index += 1
            if data_index < data_length:
                pixel[2] = int(b[:-1] + binary_data[data_index])
                data_index += 1
            if data_index >= data_length:
                break
            # print("Proses Encode = ",text)
        
    return img

def decode_text(stego):
    image = stego

    text=show_data(image)
    # text=lsb_decode(image)    
    # decrypt = go_decrypt(arr_key, text)
    return text

def show_data(image):
    binary_data = ""
    for values in image:
        for pixel in values:
            r,g,b = data2binary(pixel)
            
            binary_data += r[-1]
            binary_data += g[-1]
            binary_data += b[-1]

    all_bytes = [binary_data[i: i+8] for i in range (0,len(binary_data),8)]

    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte,2))
        if decoded_data[-5:] == "#####":
            # print("decode text : ", decoded_data)
            break 
    
    decrypt = bytes(decoded_data[:-5], encoding = 'utf-8')
    print("Hidden Object = ",decrypt)
    # decrypt = binascii.unhexlify(dec)
    return decrypt
    
# upload gambar
def upload_image(file):

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        fixname = str(id) + filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fixname))

    return fixname

# routes

@app.route('/')
def main():
    return render_template('hal_home.html')

@app.route('/encode')
def encode():
    return render_template('hal_encode.html')
 
@app.route('/encode', methods = ['GET','POST'])
def show_encode():
    if request.method == 'POST':
        if request.form['action'] == 'encode' and request.form['pesan'] != '':
            carrier = request.files['file_1'] 
            print("nama file gambar = ",carrier.filename)
            # print(cover_img.filename)
            
            filename = secure_filename(carrier.filename)
            carrier.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            carrier_img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'] + filename))
            
            dimensi = np.shape(carrier_img)
            uploads = [filename]

            data = request.form.get('pesan')
            key = request.form.get('kunci')
                # array ke bit
            arr_data = bytes(data, 'utf-8')
            a_key = bytes(key, 'utf-8')
            arr_key = pad(a_key, AES.block_size)

            nonce, cipherText = go_encrypt(arr_data,arr_key)
            pickle.dump(nonce, open('nonce.sav', 'wb'))
            # loaded_nonce = pickle.load(open('nonce.sav', 'rb'))
            stego, stego2, file_name = encode_text(cipherText, carrier_img, filename)
            # mse, psnr = go_psnr(stego2,carrier_img)
            mse, psnr = PSNR(stego,carrier_img)
            # print("MSE = ",mse,",PSNR = ",psnr)

            # decrypt = decode_text(stego)
            # plaintext = go_decrypt(arr_key, nonce, decrypt)

            return render_template('hal_encode.html', nonce=nonce,
                            cipherText=cipherText,
                            uploads = uploads,
                            carrier_img = carrier_img,
                            key = key,
                            stego = stego,
                            mse = mse,
                            psnr = psnr,
                            dimensi = dimensi,
                            file_name = file_name,
                            # decrypt = decrypt,
                            # plaintext = plaintext
                            )    
        
        #gak dipake
        if request.form['action'] == 'decode':
            stego = request.files['stego']
            stegofile = secure_filename(stego.filename)
            stego.save(os.path.join(app.config['DECODE_FOLDER'], stegofile))
            stego = cv2.imread(os.path.join(app.config['DECODE_FOLDER'] + stegofile))

            key = request.form.get('kunci2')
            # array ke bit
            a_key = bytes(key, 'utf-8')
            arr_key = pad(a_key, AES.block_size)

            # loaded_nonce = pickle.load(open('nonce.sav', 'rb'))
            decrypt = decode_text(stego)
            plaintext = go_decrypt(arr_key, nonce, decrypt)
            # result = stegoLSB

            return render_template('hal_decode.html', nonce=nonce,
                        key = key,
                        decrypt = decrypt,
                        plaintext = plaintext
                        )

         
@app.route('/decode')
def decode():
    return render_template('hal_decode.html')
      
@app.route('/decode', methods = ['GET','POST'])
def show_decode():
    if request.method == 'POST':
        stego = request.files['stego']
        print(stego.filename)
        filename = secure_filename(stego.filename)
        stego.save(os.path.join(app.config['DECODE_FOLDER'], filename))
        stego_img = cv2.imread(os.path.join(app.config['DECODE_FOLDER'] + filename))

        key = request.form.get('kunci2')
        # array ke bit
        a_key = bytes(key, 'utf-8')
        arr_key = pad(a_key, AES.block_size)

        loaded_nonce = pickle.load(open('nonce.sav', 'rb'))
        hidden_object = decode_text(stego_img)
        plaintext= go_decrypt(arr_key, loaded_nonce, hidden_object)


    return render_template('hal_decode.html', hidden_object = hidden_object,
                            loaded_nonce=loaded_nonce,
                            plaintext = plaintext
                            )

if __name__ == "__main__":
    app.run(debug = True)
