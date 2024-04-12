import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { DELETEUSER, GETALLMESSAGE, GETAllUSER, GETUSER, SENDMESSAGE } from '../../model/url';
import { User } from '../../model/user';
import { Contect } from '../../model/contect';

@Injectable({
  providedIn: 'root'
})
export class AdminService {

  constructor(private http:HttpClient) { }

  getAllUser():Observable<User[]>{
    return this.http.get<User[]>(GETAllUSER)
  }
  getuser(id:string):Observable<User>{
    return this.http.get<User>(GETUSER+id);
  }
  deleteuser(id:string):Observable<User>{
    return this.http.delete<User>(DELETEUSER+id);
  }
  getAllmessage():Observable<Contect[]>{
    return this.http.get<Contect[]>(GETALLMESSAGE);
  }
  sendMessage(id:string,res:Contect):Observable<Contect>{
    return this.http.post<Contect>(SENDMESSAGE+id,res);
  }
}
