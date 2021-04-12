"""
Datamanager for importing Semantic Scholar papers
into a MySQL database

Created on Jul 7 2019

@author: Jerónimo Arenas García

"""

import gzip
import json
import os
import re
import time
#from utils import get_size
from collections import Counter
from multiprocessing import Pool

import ipdb
import langid
import numpy as np
import pandas as pd
from tqdm import tqdm

from dbmanager.dbManager.base_dm_sql import BaseDMsql

try:
    # UCS-4
    regex = re.compile('[\U00010000-\U0010ffff]')
except re.error:
    # UCS-2
    regex = re.compile('[\uD800-\uDBFF][\uDC00-\uDFFF]')
"""
Some functions need to be defined outside the class for allowing 
parallel processing of the Semantic Scholar files. It is necessary
to do so to make pickle serialization work
"""


def ElementInList(source_list, search_string):
    if search_string in source_list:
        return 1
    else:
        return 0


def process_paper(paperEntry):
    """This function takes a dictionary with paper information as input
    and returns a list to insert in S2papers
    """
    if 'year' in paperEntry.keys():
        paper_list = [
            paperEntry['id'],
            regex.sub(' ', paperEntry['title']),
            regex.sub(' ', paperEntry['title'].lower()),
            regex.sub(' ', paperEntry['paperAbstract']),
            '\t'.join(paperEntry['entities']),
            '\t'.join(paperEntry['fieldsOfStudy']), paperEntry['s2PdfUrl'],
            '\t'.join(paperEntry['pdfUrls']), paperEntry['year'],
            paperEntry['journalVolume'].strip(),
            paperEntry['journalPages'].strip(),
            ElementInList(paperEntry['sources'], 'DBLP'),
            ElementInList(paperEntry['sources'], 'Medline'), paperEntry['doi'],
            paperEntry['doiUrl'], paperEntry['pmid']
        ]
    else:
        paper_list = [
            paperEntry['id'],
            regex.sub(' ', paperEntry['title']),
            regex.sub(' ', paperEntry['title'].lower()),
            regex.sub(' ', paperEntry['paperAbstract']),
            '\t'.join(paperEntry['entities']),
            '\t'.join(paperEntry['fieldsOfStudy']), paperEntry['s2PdfUrl'],
            '\t'.join(paperEntry['pdfUrls']), 9999,
            paperEntry['journalVolume'].strip(),
            paperEntry['journalPages'].strip(),
            ElementInList(paperEntry['sources'], 'DBLP'),
            ElementInList(paperEntry['sources'], 'Medline'), paperEntry['doi'],
            paperEntry['doiUrl'], paperEntry['pmid']
        ]

    return paper_list


def process_paperFile(gzfile):
    """Process Semantic Scholar gzip file, and extract a list of
    journals, a list of venues, a list of fields of study, and a
    list wih paper information to save in the S2papers table
    Args:
    :param gzfil: String containing the name of the file to process

    Returns:
    A list containing 3 lists: papers in file, unique journals in file,
    unique venues in file, unique fields in file
    """
    with gzip.open(gzfile, 'rt', encoding='utf8') as f:
        papers_infile = f.read().replace('}\n{', '},{')

    papers_infile = json.loads('[' + papers_infile + ']')

    # We extract venues and journals, getting rid of repetitions
    thisfile_venues = [el['venue'] for el in papers_infile]
    thisfile_venues = list(set(thisfile_venues))
    thisfile_journals = [el['journalName'] for el in papers_infile]
    thisfile_journals = list(set(thisfile_journals))
    # We extract all fields, and flatten before getting rid of repetitions
    # Flatenning is necessary because each paper has a list of fields
    thisfile_fields = [el['fieldsOfStudy'] for el in papers_infile]
    thisfile_fields = [item for sublist in thisfile_fields for item in sublist]
    thisfile_fields = list(set(thisfile_fields))
    """
    Entities are not included in current Semantic Scholar versions
    # We extract all entities, and flatten before getting rid of repetitions
    # Flatenning is necessary because each paper has a list of entities
    thisfile_entities = [el['entities'] for el in papers_infile]
    thisfile_entities = [item for sublist in thisfile_entities for item in sublist]
    thisfile_entities = list(set(thisfile_entities))
    """
    # We extract fields for the S2papers table
    lista_papers = [process_paper(el) for el in papers_infile]

    return [lista_papers, thisfile_venues, thisfile_journals, thisfile_fields]


def process_Citations(gzf):
    """
    This function takes a zfile with paper information as input
    and returns a list ready to insert in table
    """
    # Read json and separate papers
    with gzip.open(gzf, 'rt', encoding='utf8') as f:
        papers_infile = f.read().replace('}\n{', '},{')
        papers_infile = json.loads('[' + papers_infile + ']')
    # Process each paper
    cite_list = []
    for paperEntry in papers_infile:
        if len(paperEntry['outCitations']):
            for el in paperEntry['outCitations']:
                cite_list.append([paperEntry['id'], el])
    return cite_list


def process_Authors(gzf):
    """
    This function takes a zfile with paper information as input
    and returns a list ready to insert in table
    """
    try:
        # Read json and separate papers
        with gzip.open(gzf, 'rt', encoding='utf8') as f:
            papers_infile = f.read().replace('}\n{', '},{')
    except:
        print(f'Error with file {gzf}')
        return []

    papers_infile = json.loads('[' + papers_infile + ']')

    # Process each paper
    thisfile_authors = []
    for paperEntry in papers_infile:
        if len(paperEntry['authors']):
            for author in paperEntry['authors']:
                if len(author['ids']):
                    thisfile_authors.append(
                        (int(author['ids'][0]), author['name']))
    return thisfile_authors


class S2manager(BaseDMsql):
    def createDBschema(self):
        """
        Create DB table structure
        """
        for sql_cmd in schema:
            self._c.execute(sql_cmd)

        # Commit changes to database
        self._conn.commit()

        return

    def createDBindices(self):
        """
        Create DB table structure
        """
        for sql_cmd in indices:
            print('Creating index:', sql_cmd)
            self._c.execute(sql_cmd)

        # Commit changes to database
        self._conn.commit()

        return

    def importPapers(self, data_files, ncpu, chunksize=100000, update=False):
        """
        Import data from Semantic Scholar compressed data files
        available at the indicated location
        Only paper data will be imported
        """
        # STEP 1
        # Read and Insert paper data
        # We need to pass through all data files first to import venues, journalNames, entities
        # and fields. We populate also the S2papers table
        all_venues = []
        all_journals = []
        #all_entities = []
        all_fields = []

        print('Filling in table S2papers')

        gz_files = sorted([
            data_files + el for el in os.listdir(data_files)
            if el.startswith('s2-corpus')
        ])

        S2_to_ID = self.S22ID('S2papers',
                              'S2paperID',
                              'paperID',
                              chunksize=chunksize)
        venue_dict = self.S22ID('S2venues',
                                'venueName',
                                'venueID',
                                chunksize=chunksize)
        journal_dict = self.S22ID('S2journals',
                                  'journalName',
                                  'journalID',
                                  chunksize=chunksize)
        field_dict = self.S22ID('S2fields',
                                'fieldName',
                                'fieldID',
                                chunksize=chunksize)

        # We sort data in alphabetical order and insert in table
        def normalize(data):
            data = list(map(lambda x: x.lower(), data))
            data = list(set(data))
            data = [d for d in data if len(d) > 0]
            data.sort()
            return data

        if ncpu:
            # Parallel processing
            with Pool(ncpu) as p:
                with tqdm(total=len(gz_files)) as pbar:
                    for file_data in p.imap(process_paperFile, gz_files):
                        print()
                        pbar.update()

                        #Update dictionary
                        if update:
                            if S2_to_ID:
                                min_value = S2_to_ID[list(S2_to_ID.keys())[-1]]
                            else:
                                min_value = 0
                            aux_dict = self.S22ID('S2papers',
                                                  'S2paperID',
                                                  'paperID',
                                                  min_value=min_value,
                                                  chunksize=chunksize)
                            S2_to_ID = {**S2_to_ID, **aux_dict}

                            if venue_dict:
                                min_value = venue_dict[list(
                                    venue_dict.keys())[-1]]
                            else:
                                min_value = 0
                            aux_dict = self.S22ID('S2venues',
                                                  'venueName',
                                                  'venueID',
                                                  min_value=min_value,
                                                  chunksize=chunksize)
                            venue_dict = {**venue_dict, **aux_dict}

                            if journal_dict:
                                min_value = journal_dict[list(
                                    journal_dict.keys())[-1]]
                            else:
                                min_value = 0
                            aux_dict = self.S22ID('S2journals',
                                                  'journalName',
                                                  'journalID',
                                                  min_value=min_value,
                                                  chunksize=chunksize)
                            journal_dict = {**journal_dict, **aux_dict}

                            if field_dict:
                                min_value = field_dict[list(
                                    field_dict.keys())[-1]]
                            else:
                                min_value = 0
                            aux_dict = self.S22ID('S2fields',
                                                  'fieldName',
                                                  'fieldID',
                                                  min_value=min_value,
                                                  chunksize=chunksize)
                            field_dict = {**field_dict, **aux_dict}
                            del aux_dict

                        # Populate tables with the new data
                        df = pd.DataFrame(
                            file_data[0],
                            columns=[
                                'S2paperID', 'title', 'lowertitle',
                                'paperAbstract', 'entities', 'fieldsOfStudy',
                                's2PdfUrl', 'pdfUrls', 'year', 'journalVolume',
                                'journalPages', 'isDBLP', 'isMedline', 'doi',
                                'doiUrl', 'pmid'
                            ])
                        self.upsert('S2papers',
                                    'S2paperID',
                                    'paperID',
                                    df,
                                    S2_to_ID=S2_to_ID,
                                    chunksize=chunksize,
                                    update=update)

                        all_venues = file_data[1]
                        all_journals = file_data[2]
                        all_fields = file_data[3]

                        print(
                            'Filling in tables S2venues, S2journals and S2fields'
                        )
                        df = pd.DataFrame(all_venues, columns=['venueName'])
                        self.upsert('S2venues',
                                    'venueName',
                                    'venueID',
                                    df,
                                    S2_to_ID=venue_dict,
                                    chunksize=chunksize,
                                    update=update)
                        df = pd.DataFrame(all_journals,
                                          columns=['journalName'])
                        self.upsert('S2journals',
                                    'journalName',
                                    'journalID',
                                    df,
                                    S2_to_ID=journal_dict,
                                    chunksize=chunksize,
                                    update=update)
                        df = pd.DataFrame(all_fields, columns=['fieldName'])
                        self.upsert('S2fields',
                                    'fieldName',
                                    'fieldID',
                                    df,
                                    S2_to_ID=field_dict,
                                    chunksize=chunksize,
                                    update=update)

            pbar.close()
            p.close()
            p.join()

        else:
            print()
            pbar = tqdm(total=len(gz_files))
            pbar.clear()

            for gzf in gz_files:
                pbar.update(1)

                #Update dictionary
                if update:
                    if S2_to_ID:
                        min_value = S2_to_ID[list(S2_to_ID.keys())[-1]]
                    else:
                        min_value = 0
                    aux_dict = self.S22ID('S2papers',
                                          'S2paperID',
                                          'paperID',
                                          min_value=min_value,
                                          chunksize=chunksize)
                    S2_to_ID = {**S2_to_ID, **aux_dict}

                    if venue_dict:
                        min_value = venue_dict[list(venue_dict.keys())[-1]]
                    else:
                        min_value = 0
                    aux_dict = self.S22ID('S2venues',
                                          'venueName',
                                          'venueID',
                                          min_value=min_value,
                                          chunksize=chunksize)
                    venue_dict = {**venue_dict, **aux_dict}

                    if journal_dict:
                        min_value = journal_dict[list(journal_dict.keys())[-1]]
                    else:
                        min_value = 0
                    aux_dict = self.S22ID('S2journals',
                                          'journalName',
                                          'journalID',
                                          min_value=min_value,
                                          chunksize=chunksize)
                    journal_dict = {**journal_dict, **aux_dict}

                    if field_dict:
                        min_value = field_dict[list(field_dict.keys())[-1]]
                    else:
                        min_value = 0
                    aux_dict = self.S22ID('S2fields',
                                          'fieldName',
                                          'fieldID',
                                          min_value=min_value,
                                          chunksize=chunksize)
                    field_dict = {**field_dict, **aux_dict}
                    del aux_dict

                file_data = process_paperFile(gzf)

                # Populate tables with the new data
                df = pd.DataFrame(file_data[0],
                                  columns=[
                                      'S2paperID', 'title', 'lowertitle',
                                      'paperAbstract', 'entities',
                                      'fieldsOfStudy', 's2PdfUrl', 'pdfUrls',
                                      'year', 'journalVolume', 'journalPages',
                                      'isDBLP', 'isMedline', 'doi', 'doiUrl',
                                      'pmid'
                                  ])
                self.upsert('S2papers',
                            'S2paperID',
                            'paperID',
                            df,
                            S2_to_ID=S2_to_ID,
                            chunksize=chunksize,
                            update=update)

                all_venues = file_data[1]
                all_journals = file_data[2]
                all_fields = file_data[3]

                print('Filling in tables S2venues, S2journals and S2fields')
                df = pd.DataFrame(all_venues, columns=['venueName'])
                self.upsert('S2venues',
                            'venueName',
                            'venueID',
                            df,
                            S2_to_ID=venue_dict,
                            chunksize=chunksize,
                            update=update)
                df = pd.DataFrame(all_journals, columns=['journalName'])
                self.upsert('S2journals',
                            'journalName',
                            'journalID',
                            df,
                            S2_to_ID=journal_dict,
                            chunksize=chunksize,
                            update=update)
                df = pd.DataFrame(all_fields, columns=['fieldName'])
                self.upsert('S2fields',
                            'fieldName',
                            'fieldID',
                            df,
                            S2_to_ID=field_dict,
                            chunksize=chunksize,
                            update=update)

            pbar.close()

        # all_venues = normalize(all_venues)
        # all_journals = normalize(all_journals)
        # all_fields = normalize(all_fields)

        # print('Filling in tables S2venues, S2journals and S2fields')
        # df = pd.DataFrame(all_venues, columns=['venueName'])
        # self.upsert('S2venues', 'venueName', 'venueID', df, chunksize)
        # df = pd.DataFrame(all_journals, columns=['journalName'])
        # self.upsert('S2journals', 'journalName', 'journalID', df, chunksize)
        # df = pd.DataFrame(all_fields, columns=['fieldName'])
        # self.upsert('S2fields', 'fieldName', 'fieldID', df, chunksize)
        # self.insertInTable('S2venues', 'venueName', [
        #                    [el] for el in all_venues])
        # self.insertInTable('S2journals', 'journalName', [
        #                    [el] for el in all_journals])
        # self.insertInTable('S2fields', 'fieldName', [
        #                    [el] for el in all_fields])
        """
        if len(all_entities):
            all_entities.sort()
            self.insertInTable('S2entities', 'entityname', [[el] for el in all_entities])
        """

        return

    def importCitations(self, data_files, ncpu, chunksize=100000):
        """Imports Citation information"""

        # First, we need to create a dictionary to access the paperID
        # corresponding to each S2paperID
        print('Generating S2 to ID dictionary')

        # S2_to_ID = {}
        # for df in self.readDBchunks('S2papers',
        #                             'paperID',
        #                             chunksize=chunksize,
        #                             selectOptions='paperID, S2paperID',
        #                             verbose=True):
        #     ID_to_S2_list = df.values.tolist()
        #     S2_to_ID_list = [[el[1], el[0]] for el in ID_to_S2_list]
        #     aux_dict = dict(S2_to_ID_list)
        #     S2_to_ID = {**S2_to_ID, **aux_dict}
        # del df
        S2_to_ID = self.S22ID('S2papers',
                              'S2paperID',
                              'paperID',
                              chunksize=chunksize)

        # Read files
        gz_files = sorted([
            data_files + el for el in os.listdir(data_files)
            if el.startswith('s2-corpus')
        ])

        if ncpu:
            # Parallel processing
            new_citations = pd.DataFrame()
            # aux_list = []
            with Pool(ncpu) as p:
                with tqdm(total=len(gz_files)) as pbar:
                    for cite_list in p.imap(process_Citations, gz_files):
                        aux_list = []
                        print()
                        pbar.update()
                        for cite in cite_list:
                            try:
                                aux_list.append(
                                    [S2_to_ID[cite[0]], S2_to_ID[cite[1]]])
                            except:
                                pass
                        
                        citations_df = pd.DataFrame(aux_list)
                        citations_df.columns = ['paperID1', 'paperID2']
                        # Delete from table previous information of papers
                        delete = [[val] for val in citations_df['paperID1'].values]
                        self.deleteFromTable('citations',
                                            'paperID1',
                                            delete,
                                            chunksize=chunksize)
                        # Introduce new data
                        self.insertInTable('citations', ['paperID1', 'paperID2'],
                                        citations_df.values,
                                        chunksize=chunksize,
                                        verbose=True)
            pbar.close()
            p.close()
            p.join()

        else:
            print()
            pbar = tqdm(total=len(gz_files))
            pbar.clear()

            # aux_list = []
            for gzf in gz_files:
                aux_list = []
                pbar.update(1)
                cite_list = process_Citations(gzf)

                for cite in cite_list:
                    try:
                        aux_list.append([S2_to_ID[cite[0]], S2_to_ID[cite[1]]])
                    except:
                        pass
                
                citations_df = pd.DataFrame(aux_list)
                citations_df.columns = ['paperID1', 'paperID2']

                # Delete from table previous information of papers
                delete = [[val] for val in citations_df['paperID1'].values]
                self.deleteFromTable('citations',
                                    'paperID1',
                                    delete,
                                    chunksize=chunksize)
                # Introduce new data
                self.insertInTable('citations', ['paperID1', 'paperID2'],
                                citations_df.values,
                                chunksize=chunksize,
                                verbose=True)
                

        # # Populate tables with the new data
        # citations_fdf = pd.DataFrame(aux_list)
        # new_citations = pd.concat([citations_fdf], ignore_index=True)
        # new_citations.columns = ['paperID1', 'paperID2']

        # del citations_fdf

        # print('Filling in citations...\n')
        # bar = tqdm(total=len(gz_files))

        # # Delete from table previous information of papers
        # delete = [[val] for val in new_citations['paperID1'].values]

        # self.deleteFromTable('citations',
        #                      'paperID1',
        #                      delete,
        #                      chunksize=chunksize)

        # # Introduce new data
        # self.insertInTable('citations', ['paperID1', 'paperID2'],
        #                    new_citations.values,
        #                    chunksize=chunksize,
        #                    verbose=True)

        # bar.close()

        # del S2_to_ID

        return

    # def importCitations(self, data_files, chunksize):
    #     """Imports Citation information"""

    #     # First, we need to create a dictionary to access the paperID
    #     # corresponding to each S2paperID
    #     print('Generating S2 to ID dictionary')

    #     S2_to_ID = {}

    #     for df in self.readDBchunks('S2papers',
    #                                 'paperID',
    #                                 chunksize=chunksize,
    #                                 selectOptions='paperID, S2paperID',
    #                                 verbose=True):
    #         ID_to_S2_list = df.values.tolist()
    #         S2_to_ID_list = [[el[1], el[0]] for el in ID_to_S2_list]
    #         aux_dict = dict(S2_to_ID_list)
    #         S2_to_ID = {**S2_to_ID, **aux_dict}

    #     # A pass through all data files is needed to fill in tables citations

    #     def process_Citations(paperEntry):
    #         """This function takes a dictionary with paper information as input
    #         and returns a list ready to insert in citations table
    #         """
    #         cite_list = []
    #         for el in paperEntry['outCitations']:
    #             try:
    #                 cite_list.append(
    #                     [S2_to_ID[paperEntry['id']], S2_to_ID[el]])
    #             except:
    #                 pass
    #         return cite_list

    #     # Get existing citations in database
    #     citations = []
    #     for df_DB in self.readDBchunks('citations',
    #                                    'paperID1',
    #                                    chunksize=100000,
    #                                    selectOptions='paperID1, paperID2',
    #                                    verbose=True):
    #         ID_to_S2_list = df_DB.values.tolist()
    #         citations += ID_to_S2_list
    #     citations = [tuple(el) for el in citations]
    #     citations = [list(el) for el in set(citations)]

    #     gz_files = sorted([
    #         data_files + el for el in os.listdir(data_files)
    #         if el.startswith('s2-corpus')
    #     ])
    #     print('Filling in citations...\n')
    #     bar = tqdm(total=len(gz_files))
    #     for gzf in gz_files:
    #         bar.update(1)
    #         with gzip.open(gzf, 'rt', encoding='utf8') as f:
    #             papers_infile = f.read().replace('}\n{', '},{')
    #             papers_infile = json.loads('[' + papers_infile + ']')

    #             lista_citas = []
    #             for paper in papers_infile:
    #                 lista_citas += process_Citations(paper)

    #             # Populate table with the new data
    #             # self.insertInTable('citations', [
    #             #                    'paperID1', 'paperID2'], lista_citas, chunksize=chunksize, verbose=True)

    #             values_insert = list(
    #                 filter(lambda x: x not in citations, lista_citas))
    #             # values_update = list(
    #             #     filter(lambda x: x[0] in keyintable, df.values))
    #             if len(values_insert) > 0:
    #                 self.insertInTable('citations', ['paperID1', 'paperID2'],
    #                                    values_insert,
    #                                    chunksize=chunksize,
    #                                    verbose=True)

    #     bar.close()

    #     del S2_to_ID

    #     return

    def importFields(self, data_files, chunksize):
        """Imports Journals, Volumes, and Field of Study associated to each paper"""

        # We extract venues, journals and fields as dictionaries for inserting new data in tables
        df = self.readDBtable('S2venues', selectOptions='venueName, venueID')
        venues_dict = dict(df.values.tolist())
        df = self.readDBtable('S2journals',
                              selectOptions='journalName, journalID')
        journals_dict = dict(df.values.tolist())
        df = self.readDBtable('S2fields', selectOptions='fieldName, fieldID')
        fields_dict = dict(df.values.tolist())

        # Now, we need to create a dictionary to access the paperID
        # corresponding to each S2paperID
        print('Generating S2 to ID dictionary')

        S2_to_ID = self.S22ID('S2papers',
                              'S2paperID',
                              'paperID',
                              chunksize=chunksize)

        # A pass through all data files is needed to extract the data of interest
        # and fill in the tables
        def normalize(data):
            # data = data.lower()
            if len(data) > 0:
                return data
            return

        def process_Fields(paperEntry):
            """
            This function takes a dictionary with paper information as input
            and returns lists ready to insert in the corresponding tables
            """
            fields_list = []
            for el in paperEntry['fieldsOfStudy']:
                try:
                    fields_list.append([
                        S2_to_ID[paperEntry['id']], fields_dict[normalize(el)]
                    ])
                except:
                    pass

            try:
                journal_list = [[
                    S2_to_ID[paperEntry['id']],
                    journals_dict[normalize(paperEntry['journalName'])]
                ]]
            except:
                journal_list = []

            try:
                venues_list = [[
                    S2_to_ID[paperEntry['id']],
                    venues_dict[normalize(paperEntry['venue'])]
                ]]
            except:
                venues_list = []

            return fields_list, journal_list, venues_list

        gz_files = sorted([
            data_files + el for el in os.listdir(data_files)
            if el.startswith('s2-corpus')
        ])

        print('Filling in venue, journal and field of study data...\n')
        bar = tqdm(total=len(gz_files))
        for gzf in gz_files[22:]:
            bar.update(1)
            with gzip.open(gzf, 'rt', encoding='utf8') as f:
                papers_infile = f.read().replace('}\n{', '},{')
                papers_infile = json.loads('[' + papers_infile + ']')

                lista_fields = []
                lista_journals = []
                lista_venues = []
                for paper in papers_infile:
                    # try:
                    all_lists = process_Fields(paper)
                    # except:
                    #     all_lists = [[], [], []]
                    lista_fields += all_lists[0]
                    lista_journals += all_lists[1]
                    lista_venues += all_lists[2]

                print(len(lista_fields), len(lista_journals),
                      len(lista_venues))
                # Delete from table previous information of papers
                # and insert updated info
                # FIELDS
                if len(lista_fields):
                    paper, _ = zip(*lista_fields)
                    delete = [[k] for k in set(paper)]
                    self.deleteFromTable('paperField',
                                         'paperID',
                                         delete,
                                         chunksize=chunksize)
                    self.insertInTable('paperField', ['paperID', 'fieldID'],
                                       lista_fields,
                                       chunksize=chunksize,
                                       verbose=True)
                # VENUES
                if len(lista_venues):
                    paper, _ = zip(*lista_venues)
                    delete = [[k] for k in set(paper)]
                    self.deleteFromTable('paperVenue',
                                         'paperID',
                                         delete,
                                         chunksize=chunksize)
                    self.insertInTable('paperVenue', ['paperID', 'venueID'],
                                       lista_venues,
                                       chunksize=chunksize,
                                       verbose=True)
                # JOURNALS
                if len(lista_journals):
                    paper, _ = zip(*lista_journals)
                    delete = [[k] for k in set(paper)]
                    self.deleteFromTable('paperJournal',
                                         'paperID',
                                         delete,
                                         chunksize=chunksize)
                    self.insertInTable('paperJournal',
                                       ['paperID', 'journalID'],
                                       lista_journals,
                                       chunksize=chunksize,
                                       verbose=True)
        bar.close()

        del S2_to_ID

        return

    def importAuthorsData(self,
                          data_files,
                          ncpu,
                          chunksize=100000,
                          update=False):

        print('Filling authors information')

        gz_files = [
            data_files + el for el in os.listdir(data_files)
            if el.startswith('s2-corpus')
        ]
        S2_to_ID = self.S22ID('S2authors',
                              'S2authorID',
                              'authorID',
                              chunksize=chunksize)

        # S2_to_ID = {}

        def chunks(l, n):
            '''Yields successive n-sized chunks from list l.'''
            for i in range(0, len(l), n):
                yield l[i:i + n]

        for gz_chunk in chunks(gz_files, 100):
            author_counts = []
            #Update dictionary
            if update:
                if S2_to_ID:
                    min_value = S2_to_ID[list(S2_to_ID.keys())[-1]]
                else:
                    min_value = 0
                aux_dict = self.S22ID('S2authors',
                                      'S2authorID',
                                      'authorID',
                                      min_value=min_value,
                                      chunksize=chunksize)
                S2_to_ID = {**S2_to_ID, **aux_dict}
                del aux_dict

            if ncpu:
                # Parallel processing
                with Pool(ncpu) as p:
                    with tqdm(total=len(gz_chunk)) as pbar:
                        for thisfile_authors in p.imap(process_Authors,
                                                       gz_chunk):
                            print()
                            pbar.update()
                            author_counts += thisfile_authors

                pbar.close()
                p.close()
                p.join()

            else:
                print()
                pbar = tqdm(total=len(gz_chunk))
                pbar.clear()

                for gzf in gz_chunk:
                    pbar.update(1)
                    author_counts += process_Authors(gzf)

            author_counts = Counter(author_counts)
            # We insert author data in table but we need to get rid of duplicated ids
            id_name_count = [[el[0], el[1], author_counts[el]]
                             for el in author_counts]
            df = pd.DataFrame(id_name_count,
                              columns=['S2authorID', 'name', 'counts'])
            # sort according to 'id' and then by 'counts'
            df.sort_values(by=['S2authorID', 'counts'],
                           ascending=False,
                           inplace=True)
            # We get rid of duplicates, keeping first element (max counts)
            df.drop_duplicates(subset='S2authorID', keep='first', inplace=True)
            self.upsert('S2authors',
                        'S2authorID',
                        'authorID',
                        df[['S2authorID', 'name']],
                        S2_to_ID=S2_to_ID,
                        chunksize=chunksize,
                        update=False)

    def importAuthors(self, data_files):
        """Imports Authorship information (paper-author data)"""
        """
                thisfile_authors = []
                thisfile_authors2 = []
                for el in papers_infile:
                    if len(el['authors']):
                        for author in el['authors']:
                            if len(author['ids']):
                                thisfile_authors.append((int(author['ids'][0]), author['name']))
                author_counts = author_counts + Counter(thisfile_authors)
                print('Finished processing file:', gzfile)
                #print('Number of authors (str):', len(author_counts), '(', get_size(author_counts)/1e9, 'g )')
                #print('Number of authors (int):', len(author_counts2), '(', get_size(author_counts2)/1e9, 'g )')
                #print('Max author id this far:', max([el[0] for el in author_counts2.keys()]))
                #print('Last iteration time:', time.time()-start)
                #start=time.time()

        # We insert author data in table but we need to get rid of duplicated ids
        id_name_count = [[el[0], el[1], author_counts[el]] for el in author_counts]
        df = pd.DataFrame(id_name_count, columns=['id', 'name', 'counts'])
        #sort according to 'id' and then by 'counts'
        df.sort_values(by=['id', 'counts'], ascending=False, inplace=True)
        #We get rid of duplicates, keeping first element (max counts)
        df.drop_duplicates(subset='id', keep='first', inplace=True)
        self.insertInTable('S2authors', ['authorID', 'name'], df[['id', 'name']].values.tolist(), chunksize=100000, verbose=True)

        """

        # First, we need to create a dictionary to access the paperID
        # corresponding to each S2paperID
        print('Generating S2 to ID dictionary')

        chunksize = 100000
        cont = 0

        author_to_ID = self.S22ID('S2authors',
                                  'S2authorID',
                                  'authorID',
                                  chunksize=chunksize)
        paper_to_ID = self.S22ID('S2papers',
                                 'S2paperID',
                                 'paperID',
                                 chunksize=chunksize)

        # A pass through all data files is needed to fill in table paperAuthor
        def process_Authorship(paperEntry):
            """This function takes a dictionary with paper information as input
            and returns a list ready to insert in paperAuthor
            """
            author_list = [[
                paper_to_ID[paperEntry['id']], author_to_ID[int(el['ids'][0])]
            ] for el in paperEntry['authors'] if len(el['ids'])]

            return author_list

        gz_files = [
            data_files + el for el in os.listdir(data_files)
            if el.startswith('s2-corpus')
        ]
        print('Filling in authorship information...\n')
        bar = tqdm(total=len(gz_files))
        for fileno, gzf in enumerate(gz_files):
            bar.update(1)
            with gzip.open(gzf, 'rt', encoding='utf8') as f:
                papers_infile = f.read().replace('}\n{', '},{')
                papers_infile = json.loads('[' + papers_infile + ']')

                lista_author_paper = []
                for paper in papers_infile:
                    lista_author_paper += process_Authorship(paper)

                # Populate tables with the new data
                self.insertInTable('paperAuthor', ['paperID', 'authorID'],
                                   lista_author_paper,
                                   chunksize=100000,
                                   verbose=True)
        bar.close()

        return

    def importEntities(self, data_files):
        """Imports Entities associated to each paper"""

        # First, we need to create a dictionary to access the paperID
        # corresponding to each S2paperID
        print('Generating S2 to ID dictionary')

        chunksize = 100000
        cont = 0
        S2_to_ID = {}
        df = self.readDBtable('S2papers',
                              limit=chunksize,
                              selectOptions='paperID, S2paperID',
                              filterOptions='paperID>0',
                              orderOptions='paperID ASC')
        while len(df):
            cont = cont + len(df)
            # Next time, we will read from the largest retrieved ID. This is the
            # last element of the dataframe, given that we requested an ordered df
            smallest_id = df['paperID'][0]
            largest_id = df['paperID'][len(df) - 1]
            print('Number of elements processed:', cont)
            # print('Last Id read:', largest_id)
            ID_to_S2_list = df.values.tolist()
            S2_to_ID_list = [[el[1], el[0]] for el in ID_to_S2_list]
            aux_dict = dict(S2_to_ID_list)
            S2_to_ID = {**S2_to_ID, **aux_dict}
            df = self.readDBtable('S2papers',
                                  limit=chunksize,
                                  selectOptions='paperID, S2paperID',
                                  filterOptions='paperID>' + str(largest_id),
                                  orderOptions='paperID ASC')

        # We extract also a dictionary with entities values
        df = self.readDBtable('S2entities',
                              selectOptions='entityname, entityID')
        entities_dict = dict(df.values.tolist())

        # A pass through all data files is needed to fill in table paperAuthor

        def process_Entities(paperEntry):
            """This function takes a dictionary with paper information as input
            and returns a list ready to insert in paperEntity
            """
            entities_list = [[S2_to_ID[paperEntry['id']], entities_dict[el]]
                             for el in paperEntry['entities']]

            return entities_list

        gz_files = [
            data_files + el for el in os.listdir(data_files)
            if el.startswith('s2-corpus')
        ]
        print('Filling in entities information...\n')
        bar = tqdm(total=len(gz_files))
        for fileno, gzf in enumerate(gz_files):
            bar.update(1)
            with gzip.open(gzf, 'rt', encoding='utf8') as f:
                papers_infile = f.read().replace('}\n{', '},{')
                papers_infile = json.loads('[' + papers_infile + ']')

                lista_entity_paper = []
                for paper in papers_infile:
                    lista_entity_paper += process_Entities(paper)
                lista_entity_paper = list(
                    set([tuple(el) for el in lista_entity_paper]))

                # Populate tables with the new data
                self.insertInTable('paperEntity', ['paperID', 'entityID'],
                                   lista_entity_paper,
                                   chunksize=100000,
                                   verbose=True)
        bar.close()

        del S2_to_ID

        return


"""===============================================================================
==================================================================================

         *******   *******   *      *   *******   *       *      *
         *         *         *      *   *         * *   * *     * *
         *******   *         ********   ****      *   *   *    *   *
               *   *         *      *   *         *       *   *******
         *******   *******   *      *   *******   *       *   *     *

==================================================================================
==============================================================================="""

schema = [
    """CREATE TABLE S2papers(

    paperID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    S2paperID CHAR(40),

    title TEXT,
    lowertitle TEXT,
    paperAbstract TEXT,
    entities TEXT,
    fieldsOfStudy TEXT,

    s2PdfUrl VARCHAR(77),
    pdfUrls MEDIUMTEXT,

    year SMALLINT UNSIGNED,

    journalVolume VARCHAR(300),
    journalPages VARCHAR(100),

    isDBLP TINYINT(1),
    isMedline TINYINT(1),

    doi VARCHAR(128),
    doiUrl VARCHAR(256),
    pmid VARCHAR(16),

    ESP_contri TINYINT(1),
    AIselection TINYINT(1),

    langid VARCHAR(3),
    LEMAS MEDIUMTEXT

    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_520_ci""",
    """CREATE TABLE S2authors(

	authorID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    S2authorID INT UNSIGNED,
    orcidID VARCHAR(20),
    orcidGivenName VARCHAR(40),
    orcidFamilyName VARCHAR(100),
    scopusID BIGINT(20),
    name VARCHAR(256),
    influentialCitationCount SMALLINT UNSIGNED,
    ESP_affiliation TINYINT(1)

    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_520_ci""",
    """CREATE TABLE S2entities(

    entityID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    entityName VARCHAR(120)

    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_520_ci""",
    """CREATE TABLE S2fields(

    fieldID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    fieldName VARCHAR(32)

    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_520_ci""",
    """CREATE TABLE S2venues(

    venueID MEDIUMINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    venueName VARCHAR(320)

    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_520_ci""",
    """CREATE TABLE S2journals(

    journalID MEDIUMINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    journalName VARCHAR(320)

    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_520_ci""",
    """CREATE TABLE paperAuthor(

    #ID UNSIGNED INT AUTO_INCREMENT UNIQUE FIRST,

    paperAuthorID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    paperID INT UNSIGNED,
    authorID INT UNSIGNED,

    FOREIGN KEY (paperID)  REFERENCES S2papers (paperID) ON DELETE CASCADE,
    FOREIGN KEY (authorID) REFERENCES S2authors (authorID) ON DELETE CASCADE

    )""", """CREATE TABLE paperEntity(

	paperEntityID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    paperID INT UNSIGNED,
    entityID INT UNSIGNED,

    FOREIGN KEY (paperID)  REFERENCES S2papers (paperID) ON DELETE CASCADE,
    FOREIGN KEY (entityID) REFERENCES S2entities (entityID) ON DELETE CASCADE

    )""", """CREATE TABLE paperField(

    paperFieldID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    paperID INT UNSIGNED,
    fieldID INT UNSIGNED,

    FOREIGN KEY (paperID) REFERENCES S2papers (paperID) ON DELETE CASCADE,
    FOREIGN KEY (fieldID) REFERENCES S2fields (fieldID) ON DELETE CASCADE

    )""", """CREATE TABLE paperVenue(

    paperVenueID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    paperID INT UNSIGNED,
    venueID MEDIUMINT UNSIGNED,

    FOREIGN KEY (paperID) REFERENCES S2papers (paperID) ON DELETE CASCADE,
    FOREIGN KEY (venueID) REFERENCES S2venues (venueID) ON DELETE CASCADE

    )""", """CREATE TABLE paperJournal(

    paperJournalID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    paperID INT UNSIGNED,
    journalID MEDIUMINT UNSIGNED,

    FOREIGN KEY (paperID) REFERENCES S2papers (paperID) ON DELETE CASCADE,
    FOREIGN KEY (journalID) REFERENCES S2journals (journalID) ON DELETE CASCADE

    )""", """CREATE TABLE citations(

    citationID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    paperID1 INT UNSIGNED,
    paperID2 INT UNSIGNED,

    isInfluential TINYINT(1),
    MethodIntent TINYINT(1),
    BackgrIntent TINYINT(1),
    ResultIntent TINYINT(1),

    FOREIGN KEY (paperID1) REFERENCES S2papers (paperID) ON DELETE CASCADE,
    FOREIGN KEY (paperID2) REFERENCES S2papers (paperID) ON DELETE CASCADE

    )"""
]

indices = [
    'CREATE INDEX S2id on S2papers (S2paperID)',
    'CREATE INDEX S2id on S2authors (S2authorID)',
    'CREATE INDEX paper1 on citations (paperID1)',
    'CREATE INDEX paper2 on citations (paperID2)',
    'CREATE FULLTEXT INDEX pdfurls on S2papers (pdfUrls)',
    'CREATE FULLTEXT INDEX FOS on S2papers (fieldsOfStudy)'
]
