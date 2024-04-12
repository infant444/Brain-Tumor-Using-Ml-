import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { chatroom } from '../../model/chat';
import { GETCHAT, POSTCHAT } from '../../model/url';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  constructor(private Http:HttpClient) { }

  getchats(id:string):Observable<chatroom[]>{
    return this.Http.get<chatroom[]>(GETCHAT+id);
  }
  postchats(x:chatroom):Observable<chatroom[]>{
    return this.Http.post<chatroom[]>(POSTCHAT,x);
  }
}
