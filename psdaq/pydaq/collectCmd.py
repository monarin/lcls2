#!/usr/bin/env python
"""
collectCmd - send a collection command via ZMQ

Author: Chris Ford <caf@slac.stanford.edu>
"""
import time
import zmq
import zmq.utils.jsonapi as json
import pprint
import argparse
from CollectMsg import CollectMsg

def main():

    # Define commands
    command_dict = { 'plat': CollectMsg.PLAT,
                     'alloc': CollectMsg.ALLOC,
                     'connect': CollectMsg.CONNECT,
                     'dump': CollectMsg.DUMP,
                     'die': CollectMsg.DIE,
                     'kill': CollectMsg.KILL,
                     'getstate': CollectMsg.GETSTATE }

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=command_dict.keys())
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host (default localhost)')
    parser.add_argument('-P', metavar='PARTITION', default='AMO', help='partition name (default AMO)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()

    # Prepare our context and DEALER socket
    ctx = zmq.Context()
    cmd_socket = ctx.socket(zmq.DEALER)
    cmd_socket.linger = 0
    cmd_socket.RCVTIMEO = 5000 # in milliseconds
    cmd_socket.connect("tcp://%s:%d" % (args.C, CollectMsg.router_port(args.p)))

    # Compose message
    newmsg = CollectMsg(key=command_dict[args.command])
#   newmsg['partName'] = args.P
#   newmsg['platform'] = ('%d' % args.p)

    # Send message
    newmsg.send(cmd_socket)

    # Receive reply
    try:
        cmmsg = CollectMsg.recv(cmd_socket)
    except Exception as ex:
        print(ex)
    else:
        print ("Received \"%s\"" % cmmsg.key.decode())

        if (cmmsg.body is not None) and (len(cmmsg.body) > 2):
            # JSON body
            try:
                entries = json.loads(cmmsg.body)
            except Exception as ex:
                print('E: json.loads(): %s' % ex)
            else:
                if args.v:
                    print ('Entries:')
                    pprint.pprint (entries)
                # print any error messages found
                for level,nodes in entries.items():
                    for nodeid,node in enumerate(nodes):
                        try:
                            msg = node['errorInfo']['msg']
                        except KeyError:
                            pass
                        else:
                            print('%s%d: %s' % (level, nodeid, msg))
    return

if __name__ == '__main__':
    main()
