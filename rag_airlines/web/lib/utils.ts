import {clsx,type ClassValue} from 'clsx';
import {twMerge} from 'tailwind-merge';
export const cn=(...inputs:ClassValue[])=>twMerge(clsx(inputs));
export async function api<T=Record<string,unknown>>(path:string,options?:RequestInit):Promise<T>{const response=await fetch('/api'+path,{cache:'no-store',...options});if(!response.ok){const body=await response.json().catch(()=>({detail:'Service unavailable'}));throw new Error(body.detail||body.error||`Request failed (${response.status})`);}return response.json();}
