"""
@author: Jerónimo Arenas García

Import Semantic Scholar Database to MySQL DB

"""

import argparse
import configparser
import ipdb
import time
import re
from dbmanager.S2manager import S2manager
from lemmatizer.ENlemmatizer import ENLemmatizer

# db_name = 'prueba_work'
# db_connector = 'mysql'
# db_server = 'localhost'
# db_user = 'root'
# db_password = 'root'
# db_port = 3306

# db = S2manager(db_name, db_connector, db_server,
#                db_user, db_password, db_port)

# db.createDBschema()

try:
    # UCS-4
    regex = re.compile('[\U00010000-\U0010ffff]')
except re.error:
    # UCS-2
    regex = re.compile('[\uD800-\uDBFF][\uDC00-\uDFFF]')


def clean_utf8(rawdata):
    return regex.sub(' ', rawdata)


def main(interface=False,
         resetDB=False,
         importPapers=False,
         importCitations=False,
         importFields=False,
         importAuthors=False,
         importEntities=False,
         lemmatize=False,
         lemmas_query=None):
    """
    """

    cf = configparser.ConfigParser()
    cf.read('config.cf')

    #########################
    # Configuration variables
    #
    dbUSER = cf.get('DB', 'dbUSER')
    dbPASS = cf.get('DB', 'dbPASS')
    dbSERVER = cf.get('DB', 'dbSERVER')
    dbCONNECTOR = cf.get('DB', 'dbCONNECTOR')
    dbSOCKET = cf.get('DB', 'dbSOCKET')
    dbNAME = cf.get('S2', 'dbNAME')
    ncpu = int(cf.get('S2', 'ncpu'))
    chunksize = int(cf.get('S2', 'chunksize'))

    #########################
    # Datafiles
    #
    data_files = cf.get('S2', 'data_files')

    ####################################################
    # Database connection

    if dbSOCKET:
        print('socket')
        DB = S2manager(db_name=dbNAME,
                       db_connector=dbCONNECTOR,
                       path2db=None,
                       db_server=dbSERVER,
                       db_user=dbUSER,
                       db_password=dbPASS,
                       unix_socket=dbSOCKET)
    else:
        print('tcp')
        DB = S2manager(db_name=dbNAME,
                       db_connector=dbCONNECTOR,
                       path2db=None,
                       db_server=dbSERVER,
                       db_user=dbUSER,
                       db_password=dbPASS)

    ####################################################
    # 1. If activated, display console
    print(interface)
    if interface:
        while True:
            print('\nSelect option:')
            print('1. Reset database')
            print('2. Import authors & papers from data files')
            print('3. Import citations from data files')
            print(
                '4. Import journals, volumes & Fields of Study data from data files'
            )
            print('5. Import authorship from data files')
            print('6. Import entities of each paper from data files')
            print('7. Extract lemmas for the imported papers')
            print('0. Quit')
            selection = input()

            if selection == 'a':
                # df = DB.findDuplicated('citations', 'paperID1, paperID2', showDup=True)
                df = DB.findDuplicated('S2papers', 'S2paperID')
                print(df)

            if selection == '1':
                # 2. If activated, remove and create again database tables
                print('Previous info will be deleted. Continue?\n[y]/[n]')
                selection = input()
                if selection == 'y':
                    print(
                        'Regenerating the database. Existing data will be removed.'
                    )
                    # The following method deletes all existing tables, and create them
                    # again without data.
                    DB.deleteDBtables()
                    DB.createDBschema()
                    # DB.createDBindices()
            elif selection == '2':
                # 3. If activated, authors and papers data
                # will be imported from S2 data files
                t0 = time.time()
                print('Importing papers data ...')
                DB.importPapers(data_files, ncpu, chunksize)
                t1 = time.time()

                print('Importing authors data ...')
                DB.importAuthorsData(data_files, ncpu, chunksize)
                t2 = time.time()
                print(f'paper: {t1-t0}')
                print(f'author: {t2-t1}')
                print(f'total: {t2-t0}')

            elif selection == '3':
                # 4. If activated, citations data
                # will be imported from S2 data files
                print('Importing citations data ...')
                t0 = time.time()
                DB.importCitations(data_files, ncpu, chunksize)
                print(f'Time: {time.time()-t0}')
            elif selection == '4':
                # 5. If activated, journals, volumes, and Fields of Study data
                # will be imported from S2 data files
                print('Importing journal, volume and Fields of Study data ...')
                DB.importFields(data_files, chunksize)
            elif selection == '5':
                # 6. If activated, authorship data
                # will be imported from S2 data files
                print('Importing authorship data ...')
                DB.importAuthors(data_files)
            elif selection == '6':
                # 7. If activated, entities associated to each paper
                # will be imported from S2 data files
                print('Importing entities associated to each paper ...')
                DB.importEntities(data_files)
            elif selection == '7':
                ####################################################
                # 8. If activated, will carry out lemmas extraction for the
                # imported papers
                print('Lemmatizing Titles and Abstracts ...')

                # Now we start the heavy part. To avoid collapsing the server, we will
                # read and process in chunks of N articles
                chunksize = 25000
                cont = 0
                lemmas_server = cf.get('Lemmatizer', 'server')
                stw_file = cf.get('Lemmatizer', 'stw_file')
                dict_eq_file = cf.get('Lemmatizer', 'dict_eq_file')
                POS = cf.get('Lemmatizer', 'POS')
                concurrent_posts = int(cf.get('Lemmatizer',
                                              'concurrent_posts'))
                removenumbers = cf.get('Lemmatizer', 'removenumbers') == 'True'
                keepSentence = cf.get('Lemmatizer', 'keepSentence') == 'True'

                # Initialize lemmatizer
                ENLM = ENLemmatizer(lemmas_server=lemmas_server,
                                    stw_file=stw_file,
                                    dict_eq_file=dict_eq_file,
                                    POS=POS,
                                    removenumbers=removenumbers,
                                    keepSentence=keepSentence)
                selectOptions = 'paperID, title, paperAbstract'
                if lemmas_query:
                    filterOptions = 'paperID>0 AND ' + lemmas_query
                else:
                    filterOptions = 'paperID>0'
                init_time = time.time()
                df = DB.readDBtable('S2papers',
                                    limit=chunksize,
                                    selectOptions=selectOptions,
                                    filterOptions=filterOptions,
                                    orderOptions='paperID ASC')
                while (len(df)):
                    cont = cont + len(df)

                    # Next time, we will read from the largest paperID. This is the
                    # last element of the dataframe, given that we requested an ordered df
                    largest_id = df['paperID'][len(df) - 1]
                    print('Number of articles processed:', cont)
                    print('Last Article Id read:', largest_id)

                    df['alltext'] = df['title'] + '. ' + df['paperAbstract']
                    df['alltext'] = df['alltext'].apply(clean_utf8)
                    lemasBatch = ENLM.lemmatizeBatch(
                        df[['paperID', 'alltext']].values.tolist(),
                        processes=concurrent_posts)
                    # Remove entries that where not lemmatized correctly
                    lemasBatch = [[el[0], clean_utf8(el[1])]
                                  for el in lemasBatch if len(el[1])]
                    print('Successful lemmatized documents:', len(lemasBatch))
                    DB.setField('S2papers', 'paperID', ['LEMAS'], lemasBatch)
                    if lemmas_query:
                        filterOptions = 'paperID>' + \
                            str(largest_id) + ' AND ' + lemmas_query
                    else:
                        filterOptions = 'paperID>' + str(largest_id)
                    df = DB.readDBtable('S2papers',
                                        limit=chunksize,
                                        selectOptions=selectOptions,
                                        filterOptions=filterOptions,
                                        orderOptions='paperID ASC')
                    elapsed_time = time.time() - init_time
                    print('Elapsed Time (seconds):',
                          time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            elif selection == '0':
                return

            else:
                print('Invalid option')

    # ####################################################
    # # 2. If activated, remove and create again database tables
    # if resetDB:
    #     print('Regenerating the database. Existing data will be removed.')
    #     # The following method deletes all existing tables, and create them
    #     # again without data.
    #     DB.deleteDBtables()
    #     DB.createDBschema()

    # ####################################################
    # # 3. If activated, authors and papers data
    # # will be imported from S2 data files
    # if importPapers:
    #     print('Importing papers data ...')
    #     DB.importPapers(data_files, ncpu, chunksize)

    # ####################################################
    # # 4. If activated, citations data
    # # will be imported from S2 data files
    # if importCitations:
    #     print('Importing citations data ...')
    #     DB.importCitations(data_files, chunksize)

    # ####################################################
    # # 5. If activated, journals, volumes, and Fields of Study data
    # # will be imported from S2 data files
    # if importFields:
    #     print('Importing journal, volume and Fields of Study data ...')
    #     DB.importFields(data_files, chunksize)

    # ####################################################
    # # 6. If activated, authorship data
    # # will be imported from S2 data files
    # if importAuthors:
    #     print('Importing authorship data ...')
    #     DB.importAuthors(data_files)

    # ####################################################
    # # 7. If activated, entities associated to each paper
    # # will be imported from S2 data files
    # if importEntities:
    #     print('Importing entities associated to each paper ...')
    #     DB.importEntities(data_files)

    # ####################################################
    # # 8. If activated, will carry out lemmas extraction for the
    # # imported papers
    # if lemmatize:
    #     print('Lemmatizing Titles and Abstracts ...')

    #     # Now we start the heavy part. To avoid collapsing the server, we will
    #     # read and process in chunks of N articles
    #     chunksize = 25000
    #     cont = 0
    #     lemmas_server = cf.get('Lemmatizer', 'server')
    #     stw_file = cf.get('Lemmatizer', 'stw_file')
    #     dict_eq_file = cf.get('Lemmatizer', 'dict_eq_file')
    #     POS = cf.get('Lemmatizer', 'POS')
    #     concurrent_posts = int(cf.get('Lemmatizer', 'concurrent_posts'))
    #     removenumbers = cf.get('Lemmatizer', 'removenumbers') == 'True'
    #     keepSentence = cf.get('Lemmatizer', 'keepSentence') == 'True'

    #     # Initialize lemmatizer
    #     ENLM = ENLemmatizer(lemmas_server=lemmas_server, stw_file=stw_file,
    #                         dict_eq_file=dict_eq_file, POS=POS, removenumbers=removenumbers,
    #                         keepSentence=keepSentence)
    #     selectOptions = 'paperID, title, paperAbstract'
    #     if lemmas_query:
    #         filterOptions = 'paperID>0 AND ' + lemmas_query
    #     else:
    #         filterOptions = 'paperID>0'
    #     init_time = time.time()
    #     df = DB.readDBtable('S2papers', limit=chunksize, selectOptions=selectOptions,
    #                         filterOptions=filterOptions, orderOptions='paperID ASC')
    #     while (len(df)):
    #         cont = cont+len(df)

    #         # Next time, we will read from the largest paperID. This is the
    #         # last element of the dataframe, given that we requested an ordered df
    #         largest_id = df['paperID'][len(df)-1]
    #         print('Number of articles processed:', cont)
    #         print('Last Article Id read:', largest_id)

    #         df['alltext'] = df['title'] + '. ' + df['paperAbstract']
    #         df['alltext'] = df['alltext'].apply(clean_utf8)
    #         lemasBatch = ENLM.lemmatizeBatch(df[['paperID', 'alltext']].values.tolist(),
    #                                          processes=concurrent_posts)
    #         # Remove entries that where not lemmatized correctly
    #         lemasBatch = [[el[0], clean_utf8(el[1])]
    #                       for el in lemasBatch if len(el[1])]
    #         print('Successful lemmatized documents:', len(lemasBatch))
    #         DB.setField('S2papers', 'paperID', ['LEMAS'], lemasBatch)
    #         if lemmas_query:
    #             filterOptions = 'paperID>' + \
    #                 str(largest_id) + ' AND ' + lemmas_query
    #         else:
    #             filterOptions = 'paperID>' + str(largest_id)
    #         df = DB.readDBtable('S2papers', limit=chunksize, selectOptions=selectOptions,
    #                             filterOptions=filterOptions, orderOptions='paperID ASC')
    #         elapsed_time = time.time() - init_time
    #         print('Elapsed Time (seconds):', time.strftime(
    #             "%H:%M:%S", time.gmtime(elapsed_time)))

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='interface')
    parser.add_argument('--interface', action='store_true')
    parser.add_argument(
        '--resetDB',
        action='store_true',
        help='If activated, the database will be reset and re-created')
    parser.add_argument('--importPapers',
                        action='store_true',
                        help='If activated, import author and paper data')
    parser.add_argument('--importCitations',
                        action='store_true',
                        help='If activated, import citation data')
    parser.add_argument(
        '--importFields',
        action='store_true',
        help='If activated, import journals, volumes, fields data')
    parser.add_argument('--importAuthors',
                        action='store_true',
                        help='If activated, import authorship data')
    parser.add_argument('--importEntities',
                        action='store_true',
                        help='If activated, import entities data')
    parser.add_argument('--lemmatize',
                        action='store_true',
                        help='If activated, lemmatize database')
    parser.add_argument('--lemmas_query',
                        type=str,
                        dest='lemmas_query',
                        help='Query for DB elements to lemmatize')
    parser.set_defaults(lemmas_query=None)
    args = parser.parse_args()

    main(interface=args.interface,
         resetDB=args.resetDB,
         importPapers=args.importPapers,
         importCitations=args.importCitations,
         importFields=args.importFields,
         importAuthors=args.importAuthors,
         importEntities=args.importEntities,
         lemmatize=args.lemmatize,
         lemmas_query=args.lemmas_query)
