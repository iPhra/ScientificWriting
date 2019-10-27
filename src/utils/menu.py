import os

def yesno_choice(title, callback_yes=None, callback_no=None):
    """
    FROM: RecSys2019 repo https://github.com/danmontesi/recsys2019/
    Display a choice to the user. The corresponding callback will be called in case of
    affermative or negative answers.
    :param title: text to display (e.g.: 'Do you want to ...?' )
    :param callback_yes: callback function to be called in case of 'y' answer
    :param callback_no: callback function to be called in case of 'n' answer
    Return the callback result
    """

    print()
    print(f'{title} (y/n)')
    valid_inp = ['y', 'n']

    while (True):
        inp = input()
        if inp in valid_inp:
            if inp == 'y':
                if callable(callback_yes):
                    return callback_yes()
                else:
                    return 'y'
            elif inp == 'n':
                if callable(callback_no):
                    return callback_no()
                else:
                    return 'n'
        else:
            print('Wrong choice toro ;) Retry:')


def clear():
    os.system('clear')

