
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "dd_config.h"
#include "rapidxml/rapidxml.hpp"
#include "hasp_api.h"

static void usage(void)
{
    printf("HASP license management tools\n");
    printf("Copyright (C) 4Paradigm, Inc. All rights reserved.\n\n");

    printf("Usage: \n");
    printf("    hasp_mgr u <v2c_file>/<h2h_file>\n");
    printf("        updates a Sentinel protection key/attaches a detached license\n");
    printf("    hasp_mgr i [<c2v_file>]\n");
    printf("        retrieves Sentinel protection key information, output to <c2v_file> if specified\n");
    printf("    hasp_mgr c [<id_file>]\n");
    printf("        retrieves receipient information, output to <id_file> if specified\n");
    printf("    hasp_mgr r <id_file> [<h2h_file>]\n");
    printf("        rehost a license from a Sentinel SL-AdminMode/SL-UserMode key, output to <h2h_file> if specified\n");
    printf("    hasp_mgr f [<c2v_file>]\n");
    printf("        retrieves fingerprint information, output to <c2v_file> if specified\n");
}

static char *read_file(FILE *fp)
{
    char *buffer;
    size_t   buffer_length, file_size;

    /* get the file size */
    fseek(fp, 0, SEEK_END);
    file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    buffer = (char *)malloc(file_size + 1);
    if (!buffer) {
        printf("out of memory\n");
        return 0;
    }

    buffer_length = fread(buffer, sizeof(char), file_size, fp);
    if (buffer_length <= 0) {
        printf("failed to read v2c data\n");
        free(buffer);
        return 0;
    }

    /* terminate update information */
    buffer[buffer_length] = 0;

    return buffer;
}

int main(int argc, char **argv) {
    hasp_status_t  status;
    FILE          *fp;
    char          *info = 0;

    if (argc < 2) {
        usage();
        return -1;
    }

    char hostname[1024];
    if (gethostname(hostname,sizeof(hostname))) {
        printf("get hostname failed\n");
        return -1;
    }

    if (strcmp(argv[1], "c") == 0)   /* get recipient infomation */
    {
        char *info = 0;
        std::string info_scope = std::string("<haspscope><license_manager hostname=\"") + hostname + "\" /></haspscope>";
        status = hasp_get_info(info_scope.c_str(),
                               HASP_RECIPIENT,
                               LICENSE_VENDOR_CODE, &info);

        /* check if operation was successful */
        if (status != HASP_STATUS_OK) {
            printf("hasp_get_info failed with status %u\n", status);
            return -1;
        }

        printf("====== RECIPIENT INFORMATION START ======\n");
        printf("%s\n", info);
        printf("====== RECIPIENT INFORMATION END ======\n");

        if (argc == 3) {
            fp = fopen(argv[2], "wt");
            if (!fp) {
                hasp_free(info);
                printf("could not open file %s\n", argv[2]);
                return -1;
            }
            int outlen = fprintf(fp, "%s\n", info);
            fclose(fp);
            if (outlen - 1 != strlen(info)) {
                hasp_free(info);
                printf("write recipient infomation to file failed\n");
                return -1;
            }
        }

        hasp_free(info);

        printf("\nSUCCESS\n");
        return 0;
    }
    else if (strcmp(argv[1], "r") == 0)   /* use hasp_transfer to rehost a license */
    {
        char          *info = 0;
        char          *recipient;
        char          *h2h = 0;

        if (argc < 3) {
            usage();
            return -1;
        }

        // get all key id
        std::string info_scope = std::string("<haspscope><license_manager hostname=\"") + hostname + "\" /></haspscope>";
        status = hasp_get_info(info_scope.c_str(), 
        //status = hasp_get_info("<haspscope />", 
                               "<haspformat root=\"key_info\"><hasp>"
                                   "<attribute name=\"id\" />"
                                   "<attribute name=\"key_type\" />"
                                   "<attribute name=\"rehost\" />"
                                   "<attribute name=\"production_date\" />"
                               "</hasp></haspformat>",
                               LICENSE_VENDOR_CODE, &info);
        if (status != HASP_STATUS_OK) {
            printf("get key id failed with status %u\n", status);
            return -1;
        }
        rapidxml::xml_document<> key_info_doc;
        key_info_doc.parse<0>(info);
        rapidxml::xml_node<> *key_info = key_info_doc.first_node("key_info");
        printf("License(s) binded on this machine:\n");
        int key_cnt = 0;
        std::vector<std::string> key_list;
        printf("  %3s) %20s %15s %13s %24s\n", "#", "KeyID", "KeyType", "Rehostable", "ProductionDate");
        for (auto hasp = key_info->first_node("hasp"); hasp != nullptr; hasp = hasp->next_sibling()) {
            key_cnt++;
            const char* keyid = hasp->first_attribute("id")->value();
            const char* keytype = hasp->first_attribute("key_type")->value();
            const char* rehostable = hasp->first_attribute("rehost_enduser_managed")->value();
            unsigned int ts = strtoul(hasp->first_attribute("production_date")->value(), nullptr, 10);
            char production_date[64];
            strftime(production_date, sizeof(production_date), "%Y-%m-%d %H:%M:%S %Z", localtime((time_t*)&ts));
            printf("  %3d) %20s %15s %13s %24s\n", key_cnt, keyid, keytype, rehostable, production_date);
            key_list.push_back(std::string(keyid));
        }
        hasp_free(info);
        
        // key id selection
        printf("Please choose the license to rehost, (default 1):");
        size_t key_select_idx = 1;
        char in_buff[512];
        in_buff[0] = '\0';
        fgets(in_buff, sizeof(in_buff), stdin);
        // remove \r
        if (strlen(in_buff) >= 1) {
            in_buff[strlen(in_buff) - 1] = '\0';
        }
        if (strlen(in_buff) > 0) {
            key_select_idx = strtoul(in_buff, nullptr, 10);
            if (key_select_idx < 1 || key_select_idx > key_list.size()) {
                printf("Choice %s out of range, should between 1 and %lu\n", in_buff, key_list.size());
                return -1;
            }
        }
        key_select_idx--;
        const char* keyid = key_list[key_select_idx].c_str();
        printf("Confirm rehost license with Key ID %s, y/n? (default y):", keyid);
        fgets(in_buff, sizeof(in_buff), stdin);
        // remove \r
        if (strlen(in_buff) >= 1) {
            in_buff[strlen(in_buff) - 1] = '\0';
        }
        if (strlen(in_buff) > 0) {
            if (strncmp(in_buff, "y", sizeof(in_buff)) != 0) {
                printf("Rehost aborted\n");
                return -1;
            }
        }

        /* load recipient information */
        printf("Loading id file %s ... \n", argv[2]);
        fp = fopen(argv[2], "r");
        if (!fp) {
            printf("could not open file %s\n", argv[2]);
            return -1;
        }
        recipient = read_file(fp);
        if (!recipient) {
            return -1;
        }
        fclose(fp);

        // rehost
        printf("Rehosting license with Key ID %s ...\n", keyid);
        std::string transfer_action = std::string("<rehost><hasp id=\"") + keyid + "\"/></rehost>";
        std::string transfer_scope  = std::string("<haspscope><hasp id=\"") + keyid + "\"/></haspscope>";
 
        status = hasp_transfer(transfer_action.c_str(), transfer_scope.c_str(),
                        LICENSE_VENDOR_CODE, recipient, &h2h);
        free(recipient);

        /* check if operation was successful */
        if (status != HASP_STATUS_OK) {
            printf("hasp_transfer failed with status: %u\n", status);
            return -1;
        }

        printf("====== REHOST INFORMATION START ======\n");
        printf("%s\n", h2h);
        printf("====== REHOST INFORMATION END ======\n");

        /* write h2h to stdout */
        if (argc >= 4) {
            fp = fopen(argv[3], "wt");
            if (!fp) {
                hasp_free(h2h);
                printf("could not open file %s\n", argv[2]);
                return -1;
            }
            int outlen = fprintf(fp, "%s\n", h2h);
            fclose(fp);
            if (outlen - 1 != strlen(h2h)) {
                hasp_free(h2h);
                printf("write rehost infomation to file failed\n");
                return -1;
            }
        }
        hasp_free(h2h);

        printf("\nSUCCESS\n");
        return 0;
    }
    else if (strcmp(argv[1], "u") == 0) /* use hasp_update to install a v2c/h2h */
    {
        char *buffer = 0;

        /* read update from file */
        if (argc != 3) {
            usage();
            return -1;
        }

        fp = fopen(argv[2], "r");
        if (!fp) {
            printf("could not open file %s\n", argv[2]);
            free(buffer);
            return -1;
        }

        /* read the file; this function allocates 'buffer' with 'malloc' */
        buffer = read_file(fp);
        if (!buffer) {
            return -1;
        }

        fclose(fp);

        status = hasp_update(buffer, &info);
        if (status != HASP_STATUS_OK) {
            printf("hasp_update failed with status %u\n", status);
            free(buffer);
            return -1;
        }

        /* print acknowledge data to stdout */
        if (info) {
            printf("====== UPDATE ACK START ======\n");
            printf("%s\n", info);
            printf("====== UPDATE ACK END ======\n");
        }
        hasp_free(info);

        free(buffer);
        printf("\nSUCCESS\n");
        return 0;
    }
    else if (strcmp(argv[1], "i") == 0)   /* use hasp_get_info to retrieve a c2v */
    {
        /* restrict the c2v to local Sentinel keys */
        std::string info_scope = std::string("<haspscope><license_manager hostname=\"") + hostname + "\" /></haspscope>";

        status = hasp_get_info(info_scope.c_str(),
                        HASP_UPDATEINFO,
                        LICENSE_VENDOR_CODE, &info);
        if (status != HASP_STATUS_OK) {
            printf("hasp_get_info failed with status %u\n", status);
            return -1;
        }

        printf("====== UPDATE INFORMATION START ======\n");
        printf("%s\n", info);
        printf("====== UPDATE INFORMATION END ======\n");

        /* write info to file or stdout */
        if (argc == 3) {
            fp = fopen(argv[2], "wt");
            if (!fp) {
                hasp_free(info);
                printf("Could not open file %s\n", argv[2]);
                return -1;
            }
            int outlen = fprintf(fp, "%s\n", info);
            fclose(fp);
            if (outlen - 1 != strlen(info)) {
                hasp_free(info);
                printf("write update infomation to file failed\n");
                return -1;
            }
        }
        hasp_free(info);

        printf("\nSUCCESS\n");
        return 0;
    }
    else if (strcmp(argv[1], "f") == 0)   /* use hasp_get_info to retrieve a c2v */
    {
        /* restrict the c2v to local Sentinel keys */
        std::string info_scope = std::string("<haspscope><license_manager hostname=\"") + hostname + "\" /></haspscope>";
        status = hasp_get_info(info_scope.c_str(),
                        HASP_FINGERPRINT,
                        LICENSE_VENDOR_CODE, &info);

        /* check if operation was successful */
        if (status != HASP_STATUS_OK) {
            printf("hasp_get_info failed with status %u\n", status);
            return -1;
        }

        printf("====== FINGUREPRINT INFORMATION START ======\n");
        printf("%s\n", info);
        printf("====== FINGUREPRINT INFORMATION END ======\n");

        /* write info to file or stdout */
        if (argc == 3) {
            fp = fopen(argv[2], "wt");
            if (!fp) {
                hasp_free(info);
                printf("Could not open file %s\n", argv[2]);
                return -1;
            }
            int outlen = fprintf(fp, "%s\n", info);
            fclose(fp);
            if (outlen - 1 != strlen(info)) {
                hasp_free(info);
                printf("write fingureprint infomation to file failed\n");
                return -1;
            }
        }
        hasp_free(info);

        printf("\nSUCCESS\n");
        return 0;
    }
    else
    {
        usage();
    }

    return -1;

} /* main */
