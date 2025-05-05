'''
path and command routines

Copyright: S&I Challenge 2024
'''
import sys
import os

def makeDir (name, mustBeNew):
    try:
        os.makedirs (name)
    except OSError as e:
        if e.errno != 17:
            raise e
        else:
            if mustBeNew:
                sys.stderr.write (('Directory "%s" already exists.'
                    + ' Remove this directory to proceed.\n') % name)
                sys.exit (100)
            else:
                # The directory already exists; no big deal.
                pass

def linkActual (actualPath, linkPath):
    os.symlink (os.path.realpath(actualPath), linkPath)

def linkRelative (actualPath, linkPath):
    relativeActualPath = os.path.relpath (actualPath,
        os.path.dirname (linkPath))
    os.symlink (relativeActualPath, linkPath)

def checkDirExists (name):
    if not os.path.isdir (name) is True:
        sys.stderr.write (('Directory "%s" not found.'
            + ' Exiting ...\n') % name)
        sys.exit (1)

def checkFileExists (name):
    if not os.path.exists (name) is True:
        sys.stderr.write (('File "%s" not found.'
            + ' Exiting ...\n') % name)
        sys.exit (1)

def makeCmd (description = None):
    commandPath = 'CMDs'
    step = os.path.basename (sys.argv [0])
    separator = 30 * '-' + '\n'

    makeDir (commandPath, False)
    f = open ('%s/%s.cmds.txt' % (commandPath, step), 'a')

    f.write (separator)
    if description != None:
        f.write ('# ' + description + '\n')
    f.write (' '.join (sys.argv) + '\n')
    f.write (separator)

def makeCmdPath (cmdpath, description = None):
    commandPath = os.path.join('CMDs', cmdpath)
    step = os.path.basename (sys.argv [0])
    separator = 30 * '-' + '\n'

    makeDir (commandPath, False)
    f = open ('%s/%s.cmds.txt' % (commandPath, step), 'a')

    #f.write (separator)
    if description != None:
        f.write ('# ' + description + '\n')
    f.write (' '.join (sys.argv) + '\n')
    f.write (separator)

def makeMainCmd (prog, description, *args):
    commandPath = 'CMDs'
    #step = os.path.basename (prog)
    separator = 30 * '-' + '\n'

    makeDir (commandPath, False)
    f = open ('%s/%s.cmds.txt' % (commandPath, prog), 'a')

    f.write (separator)
    if description != None:
        f.write ('# ' + description + '\n')
    f.write (' '.join (args) + '\n')
    f.write (separator)


