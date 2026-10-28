'use client';
import * as Dialog from '@radix-ui/react-dialog';
import {X} from 'lucide-react';
export function ConfirmDialog({open,onOpenChange,title,children}:{open:boolean;onOpenChange:(open:boolean)=>void;title:string;children:React.ReactNode}){return <Dialog.Root open={open} onOpenChange={onOpenChange}><Dialog.Portal><Dialog.Overlay className="dialog-overlay"/><Dialog.Content className="dialog-content"><Dialog.Title>{title}</Dialog.Title><Dialog.Description>This is a fictional refund simulation. No money will be transferred.</Dialog.Description>{children}<Dialog.Close className="dialog-close" aria-label="Close confirmation"><X size={18}/></Dialog.Close></Dialog.Content></Dialog.Portal></Dialog.Root>;}
