import ldap3

class Ldap:
    """Class for LDAP related connections/operations."""

    def __init__(self, server_uri, ldap_user, ldap_pass):
        self.server = ldap3.Server(server_uri, get_info=ldap3.ALL)
        self.conn = ldap3.Connection(self.server, user=ldap_user, password=ldap_pass, auto_bind=True)

    def who_am_i(self):
        return self.conn.extend.standard.who_am_i()

    # def get_groups(self):
    #     self.conn.search('dc=fill_this_1,dc=fill_this_2', '(objectClass=group)')
    #     return self.conn.entries

    # def get_groups_with_members(self):
    #     fq_groups = [result.entry_dn for result in ldap.get_groups()]

    #     groups_with_members = {}
    #     for group in fq_groups:
    #         self.conn.search(group, '(objectclass=group)', attributes=['sAMAccountName'])

    #         if 'sAMAccountName' in self.conn.entries[0]:
    #             groups_with_members[group] = self.conn.entries[0]['sAMAccountName'].values

    #     return groups_with_members

    # def get_members_with_groups(self):
    #     groups_with_members = self.get_groups_with_members()

    #     members_with_groups = {}
    #     for group, members in groups_with_members.items():
    #         for member in members:
    #             if not member in members_with_groups:
    #                 members_with_groups[member] = []

    #             members_with_groups[member].append(group)

    #     return members_with_groups

# def auth_main(username, password):
#     LDAP_URI = 'ldap://172.16.128.94'
#     try:
#         ldap = Ldap(LDAP_URI, username, password)
#         if ldap:
#             print('User authenticated. Welcome {0}'.format(ldap.who_am_i()))
#             return {"username":username}
#     # except ldap3.core.exceptions.LDAPBindError as bind_error:
#     #     print(str(bind_error))
#     #     return False
#     except ldap3.core.exceptions.LDAPPasswordIsMandatoryError as pwd_mandatory_error:
#         print(str(pwd_mandatory_error))
#         return False

from ldap3 import Server, Connection, ALL
from ldap3.core.exceptions import LDAPBindError, LDAPPasswordIsMandatoryError

def auth_main(username, password):
    username_lower = username.lower()
    if username_lower.endswith('@ltdic.com'):
        LDAP_URI = 'ldap://172.16.128.94'
    elif username_lower.endswith('@ltsb.com'):
        LDAP_URI = 'ldap://172.16.212.51'
    else:
        print("Username must end with @ltdic.com or @ltsb.com")
        return False

    server = Server(LDAP_URI, get_info=ALL)
    try:
        # auto_bind=True will open & bind in one step
        with Connection(server, user=username, password=password, auto_bind=True) as conn:
            print(f'User authenticated. Welcome {conn.extend.standard.who_am_i()}')
            return {"username": username}

    except LDAPBindError as bind_error:
        print(f"Bind error: {bind_error}")
        return False
    except LDAPPasswordIsMandatoryError as pwd_error:
        print(f"Password required: {pwd_error}")
        return False