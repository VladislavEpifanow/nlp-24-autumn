from .tokenizer import token_specification
import re

def test_mail_boxes():
    emails = ['danishevskiy.ns@gmail.com','testmail@gmal.com', 'test_email@gmail.com', 'test_email@gmail.com.edu.ca']
    string = 'My email is {email}, write me'

    pattern = token_specification['EMAIL']

    for mail in emails:
        matches = re.findall(pattern, string.format(email=mail))
        assert mail == matches[0]

def test_phone_number():
    phones = ['+79991237854', '8 (912) 123 1234', '8-912-123-1234', '8-912-123 1234', '+7-912-123 1234']
    string = 'My phone number is {phone_number}, call me if you want'

    pattern = token_specification['PHONE']

    for phone in phones:
        matches = re.findall(pattern, string.format(phone_number=phone))
        assert phone == matches[0]

def test_date():
    dates = ['19.06.2024', '19-06-2024', '10/12/24']
    string = 'Todat is {date} 19.02'

    pattern = token_specification['DATE']

    for date in dates:
        matches = re.finditer(pattern, string.format(date=date))

        # [print(m.group(0)) for m in matches if len(m.group(0))]
        mat = next(matches).group(0)
        assert date == mat

def test_time():
    times = ['10:29:12', '7:30 PM']
    string = 'It is {time} now'

    pattern = token_specification['TIME']

    for t in times:
        matches = re.finditer(pattern, string.format(time=t))
        assert t == next(matches).group(0)

def test_name():
    names = ['K. Willis', 'Nikita Danishevskiy']
    string = 'Hello, my name is {name}'

    pattern = token_specification['NAMINGS']

    for name in names:
        mathces = re.finditer(pattern, string.format(name=name))
        assert name == next(mathces).group(0)  
