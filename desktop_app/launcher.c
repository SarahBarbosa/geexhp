#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fail_dialog(const char *message) {
    char command[4096];
    snprintf(
        command,
        sizeof(command),
        "if command -v zenity >/dev/null 2>&1; then "
        "zenity --error --title='geeXHP could not start' --text='%s'; "
        "elif command -v kdialog >/dev/null 2>&1; then "
        "kdialog --error '%s' --title 'geeXHP could not start'; "
        "else printf 'geeXHP could not start\\n%s\\n' >&2; fi",
        message,
        message,
        message
    );
    system(command);
}

int main(void) {
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len <= 0) {
        fail_dialog("Could not resolve the launcher location.");
        return 1;
    }
    exe_path[len] = '\0';

    char dir_buf[PATH_MAX];
    strncpy(dir_buf, exe_path, sizeof(dir_buf) - 1);
    dir_buf[sizeof(dir_buf) - 1] = '\0';

    char *app_dir = dirname(dir_buf);
    char script_path[PATH_MAX];
    snprintf(script_path, sizeof(script_path), "%s/run_geexhp.sh", app_dir);

    execl("/usr/bin/env", "env", "bash", script_path, (char *)NULL);

    char message[1024];
    snprintf(
        message,
        sizeof(message),
        "Could not execute run_geexhp.sh. System error: %s",
        strerror(errno)
    );
    fail_dialog(message);
    return 1;
}
